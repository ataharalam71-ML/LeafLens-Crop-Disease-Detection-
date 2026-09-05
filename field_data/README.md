# field_data/ — your own photographs

This is where the honest number comes from. Put real tomato leaf photos here —
taken on a phone, in a real field or garden, in whatever light you had — and
`field_eval.py` will report accuracy on them next to the PlantVillage number.

## Layout

One subfolder per class, named exactly as in `config.CLASS_NAMES`. **A subset is
fine** — you will rarely have field photos of all ten, and the harness does not
require them.

```
field_data/
├── Tomato_Early_blight/
│     IMG_0413.jpg
│     IMG_0414.jpg
├── Tomato_healthy/
│     IMG_0501.jpg
├── Tomato_Late_blight/
└── Tomato_Septoria_leaf_spot/
```

Valid folder names:

| Folder name | Disease |
|---|---|
| `Tomato__Target_Spot` | Target Spot |
| `Tomato__Tomato_mosaic_virus` | Mosaic Virus |
| `Tomato__Tomato_YellowLeaf__Curl_Virus` | Yellow Leaf Curl Virus |
| `Tomato_Bacterial_spot` | Bacterial Spot |
| `Tomato_Early_blight` | Early Blight |
| `Tomato_healthy` | Healthy |
| `Tomato_Late_blight` | Late Blight |
| `Tomato_Leaf_Mold` | Leaf Mold |
| `Tomato_Septoria_leaf_spot` | Septoria Leaf Spot |
| `Tomato_Spider_mites_Two_spotted_spider_mite` | Spider Mites |

A misspelled folder is skipped with a warning naming the closest valid class,
so typos are obvious rather than silent.

## How many

50–100 total is enough to be informative. Below ~50 the harness prints a
warning and you should describe the result as indicative rather than precise.
Aim for at least 5 images per class you include.

## What to photograph

The point is to be *unlike* PlantVillage, because that is where the gap lives:

- Leaves still attached to the plant, not detached on a white sheet.
- Natural background — soil, mulch, other foliage, your hand.
- Whatever light you have: overcast, harsh midday sun, shade, partial shadow.
- A range of distances, including a few where the leaf does not fill the frame.
- A few genuinely ambiguous ones. Do not curate only the clear cases —
  filtering to the easy photos rebuilds the exact bias you are measuring.

Label each photo by what it actually is. Where you are unsure of the disease,
leave it out rather than guessing; a wrong label makes the field number worse
than having no field number.

## Optional: an "is this even a leaf" set

Make a second folder anywhere (it does not need this structure) with images that
are **not** tomato leaves at all — a hand, a wall, a cup, a different crop — and
pass it to measure outright rejection:

```bash
python field_eval.py --unknown-dir not_leaves/
python calibrate_ood.py --ood-dir not_leaves/     # adds AUROC / FPR@95TPR
```

## Run it

```bash
python field_eval.py                     # the configured INFERENCE_MODEL
python field_eval.py --model all         # every trained checkpoint
```

Outputs land in `results/`: `field_results.json`, `field_vs_lab.png`,
`field_risk_coverage.png`, and a per-model report and confusion matrix.
