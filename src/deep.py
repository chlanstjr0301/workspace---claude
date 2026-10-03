# -*- coding: utf-8 -*-
"""BL-1 공식 LSTM-Autoencoder 재현 (Plan D §4.1).

구조는 가이드북 그대로: LSTM64 -> LSTM32 -> RepeatVector -> LSTM32 -> LSTM64
                        -> TimeDistributed(Dense(n_features))

두 변형을 분리한다.
  G0 : abs 적용, naive window, offset 100, 마지막 시점 MSE, 이상 포함 임계값
       -> 재현용. 최종 후보로 선정하지 않는다.
  G1 : 원신호, gap-aware, offset 0, 전체 window MSE, 정상 전용 임계값

백엔드는 TensorFlow 가 있으면 Keras, 없으면 PyTorch 로 같은 구조를 만든다.
어느 쪽도 없으면 BL-1 을 건너뛰고 나머지 파이프라인은 그대로 실행된다 (§4.1).
"""
import numpy as np

BACKEND = None
try:                                           # pragma: no cover
    import tensorflow as tf                    # noqa: F401
    from tensorflow.keras import layers, models, optimizers
    BACKEND = "tensorflow"
except Exception:
    try:
        import torch
        import torch.nn as nn
        BACKEND = "torch"
    except Exception:
        BACKEND = None


def available():
    return BACKEND is not None


# --------------------------------------------------------------------------- #
# TensorFlow / Keras
# --------------------------------------------------------------------------- #
def _tf_model(seq, nf, hidden):
    m = models.Sequential()
    m.add(layers.LSTM(hidden[0], input_shape=(seq, nf), return_sequences=True))
    m.add(layers.LSTM(hidden[1], return_sequences=False))
    m.add(layers.RepeatVector(seq))
    m.add(layers.LSTM(hidden[1], return_sequences=True))
    m.add(layers.LSTM(hidden[0], return_sequences=True))
    m.add(layers.TimeDistributed(layers.Dense(nf)))
    return m


def _tf_fit(X, seq, nf, cfg, seed, epochs):
    tf.keras.utils.set_random_seed(seed)
    m = _tf_model(seq, nf, cfg["hidden"])
    m.compile(loss="mse", optimizer=optimizers.Adam(cfg["lr"]))
    es = tf.keras.callbacks.EarlyStopping(
        monitor="loss", min_delta=1e-5, patience=40,
        restore_best_weights=True)
    h = m.fit(X, X, epochs=epochs, batch_size=cfg["batch_size"],
              verbose=0, callbacks=[es])
    return m, len(h.history["loss"])


def _tf_recon(m, X, batch=512):
    return m.predict(X, batch_size=batch, verbose=0)


# --------------------------------------------------------------------------- #
# PyTorch
# --------------------------------------------------------------------------- #
if BACKEND == "torch":
    class _TorchAE(nn.Module):
        def __init__(self, seq, nf, hidden):
            super().__init__()
            h0, h1 = hidden
            self.seq = seq
            self.e1 = nn.LSTM(nf, h0, batch_first=True)
            self.e2 = nn.LSTM(h0, h1, batch_first=True)
            self.d1 = nn.LSTM(h1, h1, batch_first=True)
            self.d2 = nn.LSTM(h1, h0, batch_first=True)
            self.out = nn.Linear(h0, nf)

        def forward(self, x):
            x, _ = self.e1(x)
            _, (h, _) = self.e2(x)
            z = h[-1].unsqueeze(1).repeat(1, self.seq, 1)
            z, _ = self.d1(z)
            z, _ = self.d2(z)
            return self.out(z)


def _torch_fit(X, seq, nf, cfg, seed, epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = _TorchAE(seq, nf, cfg["hidden"])
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    lf = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32))
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"],
                                     shuffle=True)
    best, bad, ran = np.inf, 0, 0
    for ep in range(epochs):
        tot = 0.0
        for (xb,) in dl:
            opt.zero_grad()
            loss = lf(m(xb), xb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(xb)
        tot /= len(ds)
        ran = ep + 1
        if tot < best - 1e-5:
            best, bad = tot, 0
        else:
            bad += 1
            if bad >= 40:
                break
    m.eval()
    return m, ran


def _torch_recon(m, X, batch=1024):
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32)
            outs.append(m(xb).numpy())
    return np.concatenate(outs) if outs else np.empty_like(X)


# --------------------------------------------------------------------------- #
# 공통 API
# --------------------------------------------------------------------------- #
def fit_lstm_ae(X, cfg, seed, epochs):
    """X=(n,seq,nf) 정상 데이터로만 학습."""
    seq, nf = X.shape[1], X.shape[2]
    if BACKEND == "tensorflow":
        return _tf_fit(X, seq, nf, cfg, seed, epochs)
    if BACKEND == "torch":
        return _torch_fit(X, seq, nf, cfg, seed, epochs)
    raise RuntimeError("딥러닝 백엔드 없음")


def recon_error(m, X, mode="window"):
    """mode='last' 는 가이드북 flatten() 재현(마지막 시점만),
       mode='window' 는 전체 window 평균 MSE."""
    R = _tf_recon(m, X) if BACKEND == "tensorflow" else _torch_recon(m, X)
    if mode == "last":
        return ((X[:, -1, :] - R[:, -1, :]) ** 2).mean(axis=1)
    return ((X - R) ** 2).mean(axis=(1, 2))


def minmax_fit(X):
    lo = X.reshape(-1, X.shape[2]).min(axis=0)
    hi = X.reshape(-1, X.shape[2]).max(axis=0)
    return lo, np.maximum(hi - lo, 1e-12)


def minmax_apply(X, lo, rng):
    return (X - lo) / rng
