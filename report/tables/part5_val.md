_VisDrone val split; reference run: `baseline`_

| Run | Model | imgsz | P | R | mAP50 | mAP50-95 | Params (M) | GFLOPs | Weights (MB) | Inference (ms/img) | Train time (h) | Epochs (best) | Δ mAP50-95 | Changed vs reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | yolo26n | 640 | 0.439 | 0.344 | 0.328 | 0.181 | 2.51 | 5.9 | 5.4 | 3.05 | 3.84 | 80 (69) | +0.000 | (reference) |
| p5_yolov8n | yolov8n | 640 | 0.458 | 0.339 | 0.330 | 0.184 | 3.01 | 8.2 | 6.2 | 19.46 | 8.1 | 80 (80) | +0.003 | model=yolov8n.pt |
