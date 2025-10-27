import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _rolling_rank_kernel(
    codes: np.ndarray, values: np.ndarray, window: int, n_symbols: int
) -> np.ndarray:
    """Compute rolling percentile rank per symbol using a fixed-size ring buffer."""
    n = values.shape[0]
    result = np.empty(n, dtype=np.float32)

    buffers = np.empty((n_symbols, window), dtype=np.float64)
    counts = np.zeros(n_symbols, dtype=np.int32)
    heads = np.zeros(n_symbols, dtype=np.int32)

    for i in range(n):
        code = codes[i]
        val = values[i]

        count = counts[code]
        head = heads[code]

        if count < window:
            insert_idx = (head + count) % window
            buffers[code, insert_idx] = val
            count += 1
            counts[code] = count
        else:
            buffers[code, head] = val
            head = (head + 1) % window
            heads[code] = head

        if np.isnan(val):
            result[i] = 0.0
            continue

        current_head = heads[code]
        current_count = counts[code]
        total = 0

        for j in range(current_count):
            idx = (current_head + j) % window
            other = buffers[code, idx]
            if not np.isnan(other) and other <= val:
                total += 1

        result[i] = np.float32(total / current_count)

    return result


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    window = int(window)
    if window <= 0:
        raise ValueError("window must be a positive integer")

    df = pd.read_parquet(input_path, columns=["symbol", "Close"])
    if df.empty:
        return np.empty((0, 1), dtype=np.float32)

    codes, uniques = pd.factorize(df["symbol"], sort=False)
    n_symbols = int(uniques.size)

    codes = np.ascontiguousarray(codes.astype(np.int32, copy=False))
    values = np.ascontiguousarray(df["Close"].to_numpy(dtype=np.float64, copy=False))

    result = _rolling_rank_kernel(codes, values, window, n_symbols)
    return result.reshape(-1, 1)  # must be [N, 1]

# import sys, os
# sys.path.append(os.path.dirname(__file__))  # 确保能导入同目录下的 .pyd

# import numpy as np
# import pandas as pd
# import rolling_rank_cpp

# def ops_rolling_rank(input_path: str, window: int = 20):
#     window = int(window)
#     if window <= 0:
#         raise ValueError("window must be a positive integer")

#     df = pd.read_parquet(input_path, columns=["symbol", "Close"])
#     if df.empty:
#         return np.empty((0, 1), dtype=np.float32)

#     df["symbol"] = df["symbol"].astype("category")
#     codes  = df["symbol"].cat.codes.to_numpy(np.int32,   copy=False)
#     values = df["Close"].to_numpy(          np.float32,  copy=False)  # ✅ float32 对齐 C++
#     # 可确保 C 连续：
#     # codes  = np.ascontiguousarray(codes,  dtype=np.int32)
#     # values = np.ascontiguousarray(values, dtype=np.float32)

#     n_symbols = len(df["symbol"].cat.categories)

#     res = rolling_rank_cpp.rolling_rank(codes, values, window, int(n_symbols))
#     return res.reshape(-1, 1).astype(np.float32, copy=False)

