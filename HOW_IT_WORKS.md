# How It Works — Plain English Guide
### Grounded SAM 2 · Erase · Instance Grouping

---

## Bird's-eye view

```
Your image + text prompt
        │
        ▼
┌───────────────────┐
│  Grounding DINO   │  "Find where sofa, pillow, wall etc. are"
│  (detection)      │  → outputs bounding boxes + labels + scores
└────────┬──────────┘
         │  bounding boxes
         ▼
┌───────────────────┐
│     SAM 2         │  "Draw exact pixel masks inside each box"
│  (segmentation)   │  → outputs per-pixel masks
└────────┬──────────┘
         │  masks + labels
         ▼
┌───────────────────┐
│ Instance Grouping │  "Pillow is ON sofa → merge into sofa mask"
│  (post-process)   │  → fewer, smarter masks
└────────┬──────────┘
         │  grouped masks
         ▼
┌───────────────────┐
│  Erase + Inpaint  │  "Remove sofa and fill the gap"
│  (optional)       │  → clean output image
└────────┬──────────┘
         │
         ▼
   Outputs returned
```

---

## Step 1 — Grounding DINO (Object Detection)

**What it does:** Finds WHERE objects are in the image, as bounding boxes.

**How it works:**
- You pass a text prompt like `"sofa, pillow, wall, ceiling"`
- Grounding DINO is a transformer model trained on hundreds of millions of
  image-text pairs. It jointly reads both your image and your text at the same time.
- It finds regions in the image that visually match each word in your prompt.
- Output: a set of bounding boxes, one per detected object, each labelled with
  the matching word from your prompt and a confidence score.

**Key parameter — box_threshold (default 0.35):**
- This is the minimum confidence score an object needs to be kept.
- Lower (e.g. 0.20) → detects more objects, but may include false positives.
- Higher (e.g. 0.55) → only keeps very confident detections, may miss things.

**What it does NOT do:**
- It does not draw pixel masks. It only finds rough rectangular regions.
- It does not know anything about the spatial relationship between objects
  (e.g. pillow ON sofa). That comes later.

---

## Step 2 — SAM 2 (Pixel-level Segmentation)

**What it does:** Takes each bounding box from Step 1 and draws an exact
pixel-level mask around the object inside it.

**How it works:**
- SAM 2 (Segment Anything Model 2) by Meta was trained on over 1 billion masks.
- It does NOT understand object names or categories at all.
- It only asks: "given this rectangular region, what are the pixels that form
  one coherent visual object?"
- It uses visual cues — color, texture, edges, depth — to trace the object boundary.
- This is why it can mask a pillow perfectly even though it has no idea what a
  pillow is called.

**Key parameter — mask_threshold (default 0.5):**
- Controls how tight the mask is drawn around object edges.
- Lower (e.g. 0.3) → slightly looser mask, captures more of the object edges.
- Higher (e.g. 0.7) → tighter mask, may cut off thin edges like chair legs.

**Key parameter — segmentation_threshold (default 0.5):**
- Filters out detections from Step 1 that scored below this before SAM 2 runs.
- Think of it as a quality gate — only objects Grounding DINO is reasonably
  confident about get sent to SAM 2.

---

## Step 3 — Instance Grouping (Child → Parent Merging)

**What it does:** Merges objects that are physically ON or INSIDE a parent object
into the parent's mask, so they are treated as one unit.

**The problem it solves:**
Without grouping, every object is independent. A sofa with 3 pillows would return
4 separate masks: sofa, pillow, pillow, pillow. For most furniture use cases,
you want: 1 mask for "sofa + everything on it".

**How it works — step by step:**

1. After SAM 2 runs, we have a list of (mask, label) pairs.
2. We check every detected object against a lookup table of parent → child rules:
   ```
   "sofa"  → can absorb: pillow, cushion, throw, blanket, remote, bolster
   "table" → can absorb: plate, bowl, cup, vase, pot, book, laptop, ...
   "wall"  → can absorb: painting, frame, mirror, clock, switch, socket
   ... (full list is in DEFAULT_GROUPING_RULES in predict.py)
   ```
3. For each potential child detected, we measure pixel overlap:
   ```
   overlap_ratio = (pixels in BOTH child mask AND parent mask) / (pixels in child mask)
   ```
4. If overlap_ratio >= grouping_overlap_threshold (default 0.45 = 45%):
   - The child's pixels are merged INTO the parent mask
   - The child is removed from the output list
   - A log entry records what was merged and the overlap ratio
5. If overlap_ratio < threshold: child is kept as its own separate mask.

**Why pixel overlap instead of just bbox overlap:**
Bbox overlap is imprecise. A lamp standing NEXT TO a table might have its bbox
partially overlapping the table bbox, but its actual pixels do not overlap the
table mask. Pixel overlap is much more accurate and avoids false merges.

**Key parameter — grouping_overlap_threshold (default 0.45):**
- 0.45 means "45% of the child's pixels must be inside the parent mask".
- Lower (e.g. 0.25) → more aggressive merging, good if objects are slightly
  hanging off the parent edge (e.g. a pillow half-off the sofa).
- Higher (e.g. 0.70) → stricter, only merges when child is clearly on top.

**Key parameter — custom_grouping_rules:**
- You can add extra parent → child rules at runtime without changing the code.
- Format (JSON): `{"bench": ["bag", "hat"], "countertop": ["appliance", "jar"]}`
- These are ADDED on top of the built-in rules, not replacing them.

**enable_grouping (default True):**
- Set to False to get raw independent masks for every detected object.
- Useful if you want to see what individual objects were found before grouping.

**What gets logged in metadata_json:**
```json
"grouping": {
  "enabled": true,
  "overlap_threshold": 0.45,
  "merges": [
    {
      "action": "merged",
      "child_label": "pillow",
      "child_id": 3,
      "parent_label": "sofa",
      "parent_id": 0,
      "overlap_ratio": 0.812
    }
  ]
}
```

---

## Step 4 — Erase + Fill (Optional)

**What it does:** Removes selected objects from the image and fills the gap.

**Two ways to select what to erase:**

### A) By label (erase_labels)
- Provide comma-separated labels matching words in your prompt.
- E.g. `erase_labels = "sofa, chair"`
- All masks with that label are combined into a single erase region.
- Because grouping happened first, erasing "sofa" also erases all pillows/throws
  that were merged into it.

### B) By bounding box (erase_bbox)
- Provide pixel coordinates: `"x1,y1,x2,y2"` e.g. `"100,200,500,600"`
- Everything inside that rectangle is erased, regardless of what object it is.
- Can be combined with erase_labels in the same request.

**Mask dilation (erase_mask_dilation, default 8px):**
- Before filling, the erase mask is expanded outward by this many pixels.
- This ensures object edges (which may not be perfectly masked) are also covered.
- Higher value = cleaner edges, but erases a little more of the background.
- Set to 0 to use the exact mask boundary.

---

## Step 4a — Fill Methods

### inpaint (default, recommended)
- Uses **LaMa** (Large Mask inpainting model by Samsung Research).
- LaMa was trained specifically to fill large missing regions in images
  in a way that looks natural and continues the background texture.
- It works well for walls, floors, and plain surfaces.
- It is the slowest fill method (adds ~1-2 seconds) but produces the best result.
- Works best when the object being erased is in front of a consistent background
  (e.g. a sofa against a wall).

### blur
- Applies a heavy Gaussian blur over the erased region.
- Much faster than inpainting.
- Good when you only need to obscure an area (privacy, rough mockup).
- Does NOT reconstruct what is behind the object — just smears pixels.
- Blur radius scales with erase_mask_dilation to keep edges consistent.

### color
- Fills the erased region with a solid flat color.
- Fastest option, zero compute.
- Use for: creating silhouettes, generating masks for external tools,
  or when you will composite something else into the gap later.
- erase_color: `"R,G,B"` e.g. `"255,255,255"` for white, `"0,0,0"` for black.

---

## Outputs explained

| Output | Always returned? | Description |
|---|---|---|
| original_image | Yes | Your exact input, unchanged |
| annotated_image | Yes | Coloured masks + bboxes + labels drawn on image. Grouped objects share one colour. |
| erased_image | Only if erase was requested | Image with selected regions filled using your chosen method |
| metadata_json | Yes | Full JSON with all mask data, grouping log, erase log |

### metadata_json structure
```json
{
  "inference_time_s": 2.4,
  "image_size": { "width": 1920, "height": 1080 },
  "num_masks": 4,
  "masks": [
    {
      "id": 0,
      "label": "sofa",
      "score": 0.872,
      "bbox": {
        "x1": 120.0, "y1": 300.0, "x2": 860.0, "y2": 750.0,
        "width": 740.0, "height": 450.0
      },
      "area_px": 285400
    }
  ],
  "grouping": {
    "enabled": true,
    "overlap_threshold": 0.45,
    "merges": [
      {
        "action": "merged",
        "child_label": "pillow",
        "child_id": 3,
        "parent_label": "sofa",
        "parent_id": 0,
        "overlap_ratio": 0.812
      }
    ]
  },
  "erase": {
    "performed": true,
    "fill_method": "inpaint",
    "log": [
      { "erased_label": "sofa", "mask_id": 0 }
    ]
  }
}
```

---

## All inputs at a glance

| Input | Type | Default | Purpose |
|---|---|---|---|
| image | image | — | Your room/furniture photo |
| prompt | string | "sofa, pillow..." | What to detect. Include BOTH parents AND children. |
| box_threshold | float 0-1 | 0.35 | Detection confidence. Lower = more objects found. |
| mask_threshold | float 0-1 | 0.50 | Mask edge tightness. Higher = tighter boundary. |
| min_masks | int | 1 | Floor on how many masks to return. |
| max_masks | int | 20 | Cap on how many masks to return (best kept). |
| segmentation_threshold | float 0-1 | 0.50 | Quality gate before SAM 2 runs. |
| enable_grouping | bool | true | Merge child objects into parent masks. |
| grouping_overlap_threshold | float 0-1 | 0.45 | How much child must overlap parent to be merged. |
| custom_grouping_rules | JSON string | "" | Extra parent→child rules on top of defaults. |
| erase_labels | string | "" | Labels to erase. E.g. "sofa, chair". |
| erase_bbox | string | "" | Manual erase region. E.g. "100,200,500,600". |
| erase_fill | choice | "inpaint" | How to fill: inpaint / blur / color. |
| erase_color | string | "255,255,255" | RGB fill color when erase_fill=color. |
| erase_mask_dilation | int | 8 | Pixels to expand erase mask for cleaner edges. |

---

## Models used and what they are

| Model | Made by | Size | Purpose |
|---|---|---|---|
| Grounding DINO SwinB | IDEA Research | ~340MB | Open-vocabulary object detection from text |
| SAM 2 Large | Meta AI | ~900MB | Pixel-level segmentation from bounding boxes |
| LaMa (big-lama) | Samsung Research | ~200MB | Large-area image inpainting |

All weights download automatically on first container startup. Nothing to manage manually.

---

## Prompt writing tips for furniture/interiors

**Always include children in the prompt if you want them detected:**
```
# Good — sofa AND its children are detectable
prompt = "sofa, pillow, cushion, throw, chair, wall, ceiling, table, vase, pot"

# Bad — pillows will never be detected or grouped because they weren't asked for
prompt = "sofa, chair, wall, ceiling, table"
```

**Use specific terms for better accuracy:**
```
"false ceiling" instead of just "ceiling"
"wooden dining table" instead of just "table" in complex scenes
"accent chair" to distinguish from dining chairs
```

**Lower box_threshold if objects are being missed:**
```
box_threshold = 0.25   # more permissive, picks up smaller/partially visible objects
```

**Raise segmentation_threshold if you're getting too many false detections:**
```
segmentation_threshold = 0.65   # stricter, only keeps high-confidence objects
```
