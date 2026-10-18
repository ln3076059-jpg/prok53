# 07. Causal Motion Encoding for Streaming Telematics

## 1. Strict Causality Contract
In online automotive monitoring, future frames $t+1, t+2, \dots$ do not exist. Non-causal temporal models (e.g., standard bidirectional transformers, 3D convolutions with future padding) introduce severe lookahead bias.
`CausalMotionEncoder` computes frame differentials strictly over past history:
$$\Delta I_t = |I_t - I_{t-k}|, \quad k \ge 1$$

## 2. Implementation in Roadwatch
Implemented in `training/dmd/causal_motion.py`:
* Memory-bounded FIFO frame buffer (`maxlen=8`).
* Spatial residual standardization with rolling mean/variance.
* Stream reset guards on sequence interruption.
