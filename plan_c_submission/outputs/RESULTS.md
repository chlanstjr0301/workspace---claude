# Plan C execution result

Generated: 2026-10-04T01:35:16.038466+09:00  
Mode: `full`  
Selected model: **B1_IF**

## Frozen-protocol result

- Mean window F1: 0.9492
- Mean window recall: 0.9358
- Mean AP: 0.9936
- Worst normal-block alarm events: 2
- Mean anomaly-burst detection: 0.9255
- Frozen operational window: 10 samples
- Best sensitivity-only window: 20 samples (not promoted post hoc)

Selection first minimized worst-block false-alarm events, then mean alarm events,
then recall and AP. This is why B1_IF is deployed instead of the numerically
highest-F1 model.

## Native hardware decision

The run used the laptop's native CPU (Ryzen AI 7 PRO 350, 8 cores/16 threads).
The Radeon 860M and AMD NPU are present, but the installed Python providers are
CPU-only. The dataset is small enough that the complete statistical run remains fast.

## Scope notes

- Statistical candidates, blocked validation, conformal audit, stress tests,
  window sensitivity, feature ablation, explanation repair, and alarm persistence are complete.
- The earlier TensorFlow LSTM-AE benchmark is kept as reference evidence, but is not
  mixed into this same-protocol table because TensorFlow is unavailable in Python 3.13.
- VUS-PR is intentionally not claimed; AP and burst/alarm metrics are reported instead.
