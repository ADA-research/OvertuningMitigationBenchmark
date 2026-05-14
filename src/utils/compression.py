import json
import math
import numpy as np

# -------------------------
# Binary bit-packing
# -------------------------
def pack_binary(arr: np.ndarray) -> bytes:
    arr = np.asarray(arr, dtype=np.uint8)
    return np.packbits(arr).tobytes()

def unpack_binary(data: bytes, original_len: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr)[:original_len]
    return bits.astype(np.uint8)

# -------------------------
# Multiclass label packing
# -------------------------
def bits_needed(K: int) -> int:
    return math.ceil(math.log2(K))

def pack_multiclass(labels: np.ndarray, K: int) -> (bytes, int):
    labels = labels.astype(np.uint32)
    b = bits_needed(K)
    bitstream = np.zeros(len(labels) * b, dtype=np.uint8)
    for i, label in enumerate(labels):
        for j in range(b):
            bitstream[i * b + (b - 1 - j)] = (label >> j) & 1
    packed = np.packbits(bitstream).tobytes()
    return packed, b

def unpack_multiclass(data: bytes, n_labels: int, bits_per_label: int) -> np.ndarray:
    bitstream = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    bitstream = bitstream[: n_labels * bits_per_label]
    labels = np.zeros(n_labels, dtype=np.uint32)
    for i in range(n_labels):
        val = 0
        for j in range(bits_per_label):
            val = (val << 1) | bitstream[i * bits_per_label + j]
        labels[i] = val
    return labels

# -------------------------
# Quantization
# -------------------------
def quantize_8bit(x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    xmin, xmax = float(x.min()), float(x.max())
    scale = (xmax - xmin) / 255 if xmax > xmin else 1e-9
    q = np.round((x - xmin) / scale).astype(np.uint8)
    return q, xmin, scale

def dequantize_8bit(q: np.ndarray, xmin: float, scale: float) -> np.ndarray:
    return (q.astype(np.float32) * scale) + xmin

# -------------------------
# High-level API
# -------------------------
def compress(arr: np.ndarray, *, labels=False, K=None):
    """
    High-level compression function.

    Cases:
    - Binary labels/preds: labels=True and K=2
    - Multiclass labels: labels=True and K>2
    - Probabilities/regression: labels=False, auto-quantize
    """
    arr = np.asarray(arr)

    if labels:
        if K == 2:
            data = pack_binary(arr)
            meta = {"type": "binary", "length": len(arr)}
            return data, meta
        else:
            data, bits_per_label = pack_multiclass(arr, K)
            meta = {
                "type": "multiclass",
                "length": len(arr),
                "bits_per_label": bits_per_label,
                "K": K,
            }
            return data, meta

    q, xmin, scale = quantize_8bit(arr)
    meta = {
        "type": "quantized",
        "xmin": xmin,
        "scale": scale,
        "shape": arr.shape,
    }
    return q.tobytes(), meta

def decompress(data: bytes, meta: dict):
    t = meta["type"]

    if t == "binary":
        return unpack_binary(data, meta["length"])

    if t == "multiclass":
        return unpack_multiclass(data, meta["length"], meta["bits_per_label"])

    if t == "quantized":
        q = np.frombuffer(data, dtype=np.uint8)
        arr = dequantize_8bit(q, meta["xmin"], meta["scale"])
        return arr.reshape(meta["shape"])

    raise ValueError(f"Unknown compression type: {t}")
