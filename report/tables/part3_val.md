_VisDrone val split; reference run: `baseline`_

| Run | Model | imgsz | P | R | mAP50 | mAP50-95 | Params (M) | GFLOPs | Weights (MB) | Inference (ms/img) | Train time (h) | Epochs (best) | Δ mAP50-95 | Changed vs reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | yolo26n | 640 | 0.439 | 0.344 | 0.328 | 0.181 | 2.51 | 5.9 | 5.4 | 3.05 | 3.84 | 80 (69) | +0.000 | (reference) |
| p3_lr005 | yolo26n | 640 | 0.425 | 0.336 | 0.316 | 0.173 | 2.51 | 5.9 | 5.4 | 3.45 | 3.64 | 80 (70) | -0.008 | lr0=0.005 |
| p3_lr02 | yolo26n | 640 | 0.447 | 0.342 | 0.331 | 0.183 | 2.51 | 5.9 | 5.4 | 3.06 | 3.54 | 80 (68) | +0.002 | lr0=0.02 |

_Inference (ms/img) and Train time (h) are as measured on each run's own device and are not comparable across runs; see results/speed/ and report §1.4._
