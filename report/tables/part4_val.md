_VisDrone val split; reference run: `p3_lr02`_

| Run | Model | imgsz | P | R | mAP50 | mAP50-95 | Params (M) | GFLOPs | Weights (MB) | Inference (ms/img) | Train time (h) | Epochs (best) | Δ mAP50-95 | Changed vs reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p3_lr02 | yolo26n | 640 | 0.447 | 0.342 | 0.331 | 0.183 | 2.51 | 5.9 | 5.4 | 3.06 | 3.54 | 80 (68) | +0.000 | (reference) |
| p4_cos_lr | yolo26n | 640 | 0.447 | 0.345 | 0.330 | 0.182 | 2.51 | 5.9 | 5.4 | 1.26 | 0.66 | 80 (67) | -0.001 | cos_lr=True |
