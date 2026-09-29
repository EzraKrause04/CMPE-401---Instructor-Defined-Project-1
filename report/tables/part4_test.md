_VisDrone test split; reference run: `p3_lr02`_

| Run | Model | imgsz | P | R | mAP50 | mAP50-95 | Params (M) | GFLOPs | Weights (MB) | Inference (ms/img) | Train time (h) | Epochs (best) | Δ mAP50-95 | Changed vs reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p3_lr02 | yolo26n | 640 | 0.406 | 0.312 | 0.280 | 0.153 | 2.51 | 5.9 | 5.4 | 2.88 | 3.54 | 80 (68) | +0.000 | (reference) |
| p4_cos_lr | yolo26n | 640 | 0.394 | 0.317 | 0.278 | 0.151 | 2.51 | 5.9 | 5.4 | 1.13 | 0.66 | 80 (67) | -0.002 | cos_lr=True |

_Inference (ms/img) and Train time (h) are as measured on each run's own device and are not comparable across runs; see results/speed/ and report §1.4._
