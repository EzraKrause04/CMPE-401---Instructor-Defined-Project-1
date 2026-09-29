_VisDrone test split; reference run: `baseline`_

| Run | Model | imgsz | P | R | mAP50 | mAP50-95 | Params (M) | GFLOPs | Weights (MB) | Inference (ms/img) | Train time (h) | Epochs (best) | Δ mAP50-95 | Changed vs reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | yolo26n | 640 | 0.387 | 0.307 | 0.270 | 0.146 | 2.51 | 5.9 | 5.4 | 3.07 | 3.84 | 80 (69) | +0.000 | (reference) |
| p3_lr005 | yolo26n | 640 | 0.381 | 0.300 | 0.265 | 0.144 | 2.51 | 5.9 | 5.4 | 3.05 | 3.64 | 80 (70) | -0.002 | lr0=0.005 |
| p3_lr02 | yolo26n | 640 | 0.406 | 0.312 | 0.280 | 0.153 | 2.51 | 5.9 | 5.4 | 2.88 | 3.54 | 80 (68) | +0.007 | lr0=0.02 |
