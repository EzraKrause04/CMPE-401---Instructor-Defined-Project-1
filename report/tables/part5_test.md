_VisDrone test split; reference run: `baseline`_

| Run | Model | imgsz | P | R | mAP50 | mAP50-95 | Params (M) | GFLOPs | Weights (MB) | Inference (ms/img) | Train time (h) | Epochs (best) | Δ mAP50-95 | Changed vs reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | yolo26n | 640 | 0.387 | 0.307 | 0.270 | 0.146 | 2.51 | 5.9 | 5.4 | 3.07 | 3.84 | 80 (69) | +0.000 | (reference) |
| p5_yolov8n | yolov8n | 640 | 0.384 | 0.297 | 0.262 | 0.145 | 3.01 | 8.2 | 6.2 | 19.93 | 8.1 | 80 (80) | -0.002 | model=yolov8n.pt |

_Inference (ms/img) and Train time (h) are as measured on each run's own device and are not comparable across runs; see results/speed/ and report §1.4._
