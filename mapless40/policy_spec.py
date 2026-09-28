"""torch 없이도 import 되는 인코더 구조 상수 (policy.py · viz_encoder.py 공용)."""

# 1D LiDAR 인코더 층: (out_ch, kernel, stride, pad).  1125 → 375 → 125 → 63
CONV1D_LAYERS = ((16, 7, 3, 3), (32, 5, 3, 2), (64, 5, 2, 2))
