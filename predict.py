import os
import json
import time
import tempfile
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from cog import BasePredictor, Input, Path as CogPath

# ── Grounding DINO ──────────────────────────────────────────────────────────
from groundingdino.util.inference import load_model as load_gdino, predict as gdino_predict
from groundingdino.util import box_ops

# ── SAM 2 ───────────────────────────────────────────────────────────────────
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ---------------------------------------------------------------------------
GDINO_CONFIG       = "/src/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
GDINO_WEIGHTS_URL  = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    "v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
)
GDINO_WEIGHTS_PATH = "/weights/groundingdino_swinb.pth"
SAM2_CHECKPOINT    = "/weights/sam2_hiera_large.pt"
SAM2_CONFIG        = "sam2_hiera_l.yaml"
LAMA_URL           = "https://huggingface.co/JosephCatrambone/big-lama-torchscript/resolve/main/lama.pt"
LAMA_PATH          = "/weights/big-lama.pt"
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

# ── Default parent → child grouping rules ───────────────────────────────────
DEFAULT_GROUPING_RULES: dict = {
    "sofa":             ["pillow", "cushion", "throw", "blanket", "remote", "bolster"],
    "couch":            ["pillow", "cushion", "throw", "blanket", "remote", "bolster"],
    "armchair":         ["pillow", "cushion", "throw"],
    "chair":            ["pillow", "cushion"],
    "bed":              ["pillow", "cushion", "blanket", "bolster", "duvet", "sheet"],
    "dining table":     ["plate", "bowl", "cup", "glass", "bottle", "vase",
                         "fruit", "pot", "book", "candle", "napkin"],
    "table":            ["plate", "bowl", "cup", "glass", "bottle", "vase",
                         "fruit", "pot", "book", "candle", "laptop", "remote",
                         "magazine", "flowers"],
    "coffee table":     ["remote", "book", "magazine", "cup", "glass",
                         "candle", "vase", "tray"],
    "desk":             ["laptop", "monitor", "keyboard", "mouse", "book",
                         "cup", "pen", "lamp", "frame"],
    "shelf":            ["book", "pot", "vase", "frame", "figurine",
                         "plant", "box", "bottle"],
    "cabinet":          ["handle", "knob"],
    "wall":             ["switch", "socket", "painting", "artwork",
                         "frame", "mirror", "clock", "lamp"],
    "kitchen counter":  ["pot", "bowl", "plate", "knife", "cutting board",
                         "bottle", "cup", "appliance"],
}


def download_if_missing(url: str, dest: str):
    if not os.path.exists(dest):
        print(f"Downloading {url} → {dest}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(url, dest)


# ── LaMa inpainting ─────────────────────────────────────────────────────────
def lama_inpaint(img_np: np.ndarray, mask_np: np.ndarray, lama_model) -> np.ndarray:
    """
    img_np  : H x W x 3  uint8 RGB
    mask_np : H x W       uint8 (255 = region to fill in)
    returns : H x W x 3  uint8 RGB inpainted result
    """
    H, W   = img_np.shape[:2]
    pad_h  = (8 - H % 8) % 8
    pad_w  = (8 - W % 8) % 8
    img_p  = np.pad(img_np,  ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    mask_p = np.pad(mask_np, ((0, pad_h), (0, pad_w)),          mode="reflect")

    img_t  = torch.from_numpy(img_p).permute(2,0,1).float().div(255).unsqueeze(0).to(DEVICE)
    mask_t = torch.from_numpy(mask_p).float().div(255).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = lama_model(img_t, mask_t)

    result = out[0].permute(1,2,0).cpu().numpy()
    result = (result * 255).clip(0, 255).astype(np.uint8)
    return result[:H, :W]


# ── Instance grouping ────────────────────────────────────────────────────────
def apply_grouping(masks, boxes, labels, scores, rules, overlap_threshold=0.45):
    """
    Merges child object masks into their parent when the child's pixels
    overlap sufficiently with the parent mask.

    overlap_threshold: fraction of child pixels that must lie inside
                       the parent mask to trigger a merge.
    """
    N        = len(labels)
    absorbed = [False] * N
    log      = []

    label_to_idx = {}
    for i, lbl in enumerate(labels):
        label_to_idx.setdefault(lbl.lower(), []).append(i)

    for parent_label, child_labels in rules.items():
        for pidx in label_to_idx.get(parent_label.lower(), []):
            if absorbed[pidx]:
                continue
            parent_mask = masks[pidx].astype(bool)
            if parent_mask.sum() == 0:
                continue

            for child_label in child_labels:
                for cidx in label_to_idx.get(child_label.lower(), []):
                    if absorbed[cidx] or cidx == pidx:
                        continue
                    child_mask = masks[cidx].astype(bool)
                    child_area = child_mask.sum()
                    if child_area == 0:
                        continue

                    overlap_ratio = np.logical_and(parent_mask, child_mask).sum() / child_area

                    if overlap_ratio >= overlap_threshold:
                        masks[pidx]  = np.logical_or(parent_mask, child_mask)
                        parent_mask  = masks[pidx].astype(bool)
                        absorbed[cidx] = True
                        log.append({
                            "action":        "merged",
                            "child_label":   labels[cidx],
                            "child_id":      cidx,
                            "parent_label":  labels[pidx],
                            "parent_id":     pidx,
                            "overlap_ratio": round(float(overlap_ratio), 3),
                        })

    keep       = [i for i in range(N) if not absorbed[i]]
    return masks[keep], boxes[keep], [labels[i] for i in keep], scores[keep], log


class Predictor(BasePredictor):

    def setup(self):
        print("Loading Grounding DINO ...")
        download_if_missing(GDINO_WEIGHTS_URL, GDINO_WEIGHTS_PATH)
        self.gdino = load_gdino(GDINO_CONFIG, GDINO_WEIGHTS_PATH)
        self.gdino.eval().to(DEVICE)

        print("Loading SAM 2 ...")
        self.sam2 = SAM2ImagePredictor(
            build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
        )

        print("Loading LaMa inpainting model ...")
        download_if_missing(LAMA_URL, LAMA_PATH)
        self.lama = torch.jit.load(LAMA_PATH, map_location=DEVICE).eval()

        print("All models ready.")

    def predict(
        self,
        image: CogPath = Input(
            description="Input image (JPEG/PNG)."
        ),
        prompt: str = Input(
            description=(
                "Comma-separated objects to detect and segment. "
                "Include BOTH parents and children you want detected. "
                "E.g. 'sofa, pillow, cushion, chair, wall, ceiling, table, vase, pot'"
            ),
            default="sofa, pillow, cushion, chair, wall, ceiling, table, vase"
        ),

        # ── Segmentation ────────────────────────────────────────────────────
        box_threshold: float = Input(
            description="Detection confidence threshold (0-1). Lower = more detections, more noise.",
            default=0.35, ge=0.05, le=0.95
        ),
        mask_threshold: float = Input(
            description="SAM 2 mask tightness (0-1). Higher = tighter masks.",
            default=0.5, ge=0.0, le=1.0
        ),
        min_masks: int = Input(
            description="Minimum number of masks to return.",
            default=1, ge=1, le=50
        ),
        max_masks: int = Input(
            description="Maximum number of masks to return (top-scoring kept).",
            default=20, ge=1, le=50
        ),
        segmentation_threshold: float = Input(
            description="Minimum detection score to keep (0-1).",
            default=0.5, ge=0.0, le=1.0
        ),

        # ── Grouping ────────────────────────────────────────────────────────
        enable_grouping: bool = Input(
            description=(
                "Merge child objects into their parent. "
                "E.g. pillows on a sofa become part of the sofa mask."
            ),
            default=True
        ),
        grouping_overlap_threshold: float = Input(
            description=(
                "Fraction of child mask pixels that must overlap parent mask to trigger merge. "
                "0.45 = 45 percent. Lower = more aggressive merging."
            ),
            default=0.45, ge=0.1, le=0.95
        ),
        custom_grouping_rules: str = Input(
            description=(
                "Optional JSON to add extra grouping rules on top of defaults. "
                "Format: {\"sofa\": [\"toy\"], \"table\": [\"fruit\"]}. "
                "Leave empty to use built-in rules only."
            ),
            default=""
        ),

        # ── Output toggles ─────────────────────────────────────────────────
        output_masks: bool = Input(
            description="Generate individual segmentation masks.",
            default=True
        ),
        output_segmentation: bool = Input(
            description="Generate annotated image with colored overlays and labels.",
            default=True
        ),
        output_bbox: bool = Input(
            description="Include bounding box data in metadata.",
            default=True
        ),
        output_eraser: bool = Input(
            description="Enable object erasing (requires erase_labels or erase_bbox).",
            default=False
        ),

        # ── Erase ───────────────────────────────────────────────────────────
        erase_labels: str = Input(
            description=(
                "Comma-separated labels to erase from the image. "
                "E.g. 'sofa, chair'. Grouped children are erased with parent. "
                "Leave empty to skip."
            ),
            default=""
        ),
        erase_bbox: str = Input(
            description=(
                "Manually erase a rectangular region: 'x1,y1,x2,y2' in pixels. "
                "E.g. '100,200,500,600'. Leave empty to skip."
            ),
            default=""
        ),
        erase_fill: str = Input(
            description=(
                "How to fill erased regions: "
                "'inpaint' = AI fills naturally (LaMa, best quality), "
                "'blur' = gaussian blur, "
                "'color' = solid fill color."
            ),
            default="inpaint",
            choices=["inpaint", "blur", "color"]
        ),
        erase_color: str = Input(
            description="Solid fill color when erase_fill=color. Format: 'R,G,B'. E.g. '255,255,255'.",
            default="255,255,255"
        ),
        erase_mask_dilation: int = Input(
            description="Pixels to expand erase mask outward to cleanly cover object edges.",
            default=8, ge=0, le=50
        ),

    ) -> dict:
        t0 = time.time()

        # Load image
        pil_img = Image.open(str(image)).convert("RGB")
        img_np  = np.array(pil_img)
        H, W    = img_np.shape[:2]

        # Grounding DINO
        import torchvision.transforms as T
        transform  = T.Compose([
            T.Resize((800, 800)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(pil_img)
        caption    = prompt.lower().strip().rstrip(".")
        caption    = caption.replace(",", " . ") + " ."

        with torch.no_grad():
            boxes_norm, logits, phrases = gdino_predict(
                model=self.gdino,
                image=img_tensor,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=box_threshold * 0.8,
                device=DEVICE,
            )

        if len(boxes_norm) == 0:
            return self._empty_result(pil_img)

        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_norm) * torch.tensor(
            [W, H, W, H], dtype=torch.float32
        )
        scores = logits.cpu().numpy()

        keep       = scores >= segmentation_threshold
        boxes_xyxy = boxes_xyxy[keep]
        scores     = scores[keep]
        phrases    = [p for p, k in zip(phrases, keep) if k]

        if len(boxes_xyxy) == 0:
            return self._empty_result(pil_img)

        order      = np.argsort(scores)[::-1]
        n_keep     = max(min_masks, min(max_masks, len(order)))
        order      = order[:n_keep]
        boxes_xyxy = boxes_xyxy[order].cpu().numpy()
        scores     = scores[order]
        phrases    = [phrases[i] for i in order]

        # SAM 2
        self.sam2.set_image(img_np)
        all_masks, _, _ = self.sam2.predict(
            point_coords=None, point_labels=None,
            box=boxes_xyxy, multimask_output=False,
            mask_threshold=mask_threshold,
        )
        if all_masks.ndim == 4:
            all_masks = all_masks[:, 0]

        # Instance grouping
        grouping_log = []
        if enable_grouping:
            active_rules = dict(DEFAULT_GROUPING_RULES)
            if custom_grouping_rules.strip():
                try:
                    for parent, children in json.loads(custom_grouping_rules).items():
                        existing = active_rules.get(parent.lower(), [])
                        active_rules[parent.lower()] = list(set(existing + children))
                except Exception:
                    pass

            all_masks, boxes_xyxy, phrases, scores, grouping_log = apply_grouping(
                all_masks.copy(), boxes_xyxy.copy(), phrases, scores,
                active_rules, grouping_overlap_threshold,
            )

        # Build metadata
        metadata = []
        for i, (mask, box, score, label) in enumerate(
            zip(all_masks, boxes_xyxy, scores, phrases)
        ):
            entry = {
                "id":      i,
                "label":   label,
                "score":   round(float(score), 4),
                "area_px": int(mask.sum()),
            }
            if output_bbox:
                x1, y1, x2, y2 = box.tolist()
                entry["bbox"] = {
                    "x1": round(x1,1), "y1": round(y1,1),
                    "x2": round(x2,1), "y2": round(y2,1),
                    "width":  round(x2-x1,1),
                    "height": round(y2-y1,1),
                }
            metadata.append(entry)

        # Save original
        orig_path = Path(tempfile.mktemp(suffix="_original.png"))
        pil_img.save(str(orig_path))

        # Masks
        mask_paths = []
        if output_masks:
            for i, (mask, label) in enumerate(zip(all_masks, phrases)):
                mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
                mp = Path(tempfile.mktemp(suffix=f"_mask_{i}_{label}.png"))
                mask_img.save(str(mp))
                mask_paths.append(mp)

        # Annotated
        ann_path = None
        if output_segmentation:
            ann_path = self._draw_annotations(img_np, all_masks, boxes_xyxy, phrases, scores)

        # Erase
        erased_path = None
        erase_log   = []

        if output_eraser and (erase_labels.strip() or erase_bbox.strip()):
            erase_mask = np.zeros((H, W), dtype=np.uint8)

            if erase_labels.strip():
                targets = {t.strip().lower() for t in erase_labels.split(",") if t.strip()}
                for i, (mask, label) in enumerate(zip(all_masks, phrases)):
                    if label.lower() in targets:
                        erase_mask[mask > 0] = 255
                        erase_log.append({"erased_label": label, "mask_id": i})

            if erase_bbox.strip():
                try:
                    x1e,y1e,x2e,y2e = [int(v) for v in erase_bbox.split(",")]
                    erase_mask[max(0,y1e):min(H,y2e), max(0,x1e):min(W,x2e)] = 255
                    erase_log.append({"erased_bbox": {"x1":x1e,"y1":y1e,"x2":x2e,"y2":y2e}})
                except Exception:
                    erase_log.append({"error": "Could not parse erase_bbox"})

            if erase_mask_dilation > 0:
                k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (erase_mask_dilation*2+1, erase_mask_dilation*2+1)
                )
                erase_mask = cv2.dilate(erase_mask, k, iterations=1)

            erased_np = img_np.copy()
            if erase_fill == "inpaint":
                erased_np = lama_inpaint(img_np, erase_mask, self.lama)
            elif erase_fill == "blur":
                br = max(51, erase_mask_dilation * 6 + 1) | 1
                blurred = cv2.GaussianBlur(img_np, (br, br), 0)
                erased_np[erase_mask > 0] = blurred[erase_mask > 0]
            elif erase_fill == "color":
                try:
                    r,g,b = [int(c) for c in erase_color.split(",")]
                except Exception:
                    r,g,b = 255,255,255
                erased_np[erase_mask > 0] = [r, g, b]

            erased_path = Path(tempfile.mktemp(suffix="_erased.png"))
            Image.fromarray(erased_np).save(str(erased_path))

        # Metadata JSON
        meta_path = Path(tempfile.mktemp(suffix="_metadata.json"))
        meta_path.write_text(json.dumps({
            "inference_time_s": round(time.time() - t0, 2),
            "image_size":       {"width": W, "height": H},
            "num_masks":        len(metadata),
            "masks":            metadata,
            "grouping": {
                "enabled":           enable_grouping,
                "overlap_threshold": grouping_overlap_threshold,
                "merges":            grouping_log,
            },
            "erase": {
                "performed":   bool(erased_path),
                "fill_method": erase_fill if erased_path else None,
                "log":         erase_log,
            },
        }, indent=2))

        result = {
            "original_image":  CogPath(orig_path),
            "metadata_json":   CogPath(meta_path),
        }
        if ann_path:
            result["annotated_image"] = CogPath(ann_path)
        if mask_paths:
            for i, mp in enumerate(mask_paths):
                result[f"mask_{i}"] = CogPath(mp)
        if erased_path:
            result["erased_image"] = CogPath(erased_path)
        return result

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _draw_annotations(self, img_np, masks, boxes, labels, scores):
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.imshow(img_np)
        ax.axis("off")
        cmap = plt.get_cmap("tab20")
        for i, (mask, box, label, score) in enumerate(zip(masks, boxes, labels, scores)):
            c = cmap(i % 20)[:3]
            overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
            overlay[mask > 0] = [*c, 0.45]
            ax.imshow(overlay)
            x1, y1, x2, y2 = box
            ax.add_patch(patches.Rectangle(
                (x1,y1), x2-x1, y2-y1, linewidth=2, edgecolor=c, facecolor="none"
            ))
            ax.text(x1, max(y1-5, 10), f"{label} {score:.2f}",
                    fontsize=9, color="white",
                    bbox=dict(facecolor=c, alpha=0.75, pad=2, edgecolor="none"))
        plt.tight_layout(pad=0)
        out = Path(tempfile.mktemp(suffix="_annotated.png"))
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    def _empty_result(self, pil_img):
        p = Path(tempfile.mktemp(suffix="_original.png"))
        pil_img.save(str(p))
        m = Path(tempfile.mktemp(suffix="_metadata.json"))
        m.write_text(json.dumps({
            "num_masks": 0, "masks": [],
            "grouping": {"merges": []},
            "erase": {"performed": False},
            "message": "No objects detected. Try lowering box_threshold or segmentation_threshold.",
        }, indent=2))
        return {"original_image": CogPath(p), "annotated_image": CogPath(p), "metadata_json": CogPath(m)}
