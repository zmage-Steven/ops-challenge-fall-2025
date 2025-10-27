import numpy as np
import pandas as pd


def _rolling_percentile_rank(window_values: np.ndarray) -> float:
    """Return percentile rank of last observation within the window."""
    length = window_values.size
    if length == 0:
        return np.nan

    current = window_values[-1]
    # NaNs compare as False, matching the original behaviour.
    rank_count = np.count_nonzero(window_values <= current)
    return rank_count / length


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    df = pd.read_parquet(input_path, columns=["symbol", "Close"])

    rolling = (
        df.groupby("symbol", sort=False)["Close"]
        .rolling(window=window, min_periods=1)
        .apply(_rolling_percentile_rank, raw=True)
        .droplevel(0)
    )

    res = rolling.to_numpy(dtype=np.float32, copy=False)
    return res.reshape(-1, 1)  # must be [N, 1]

# import numpy as np
# import pandas as pd
# from numba import njit


# @njit(cache=True)
# def _rolling_rank_kernel(
#     codes: np.ndarray, values: np.ndarray, window: int, n_symbols: int
# ) -> np.ndarray:
#     """Compute rolling percentile rank per symbol using a fixed-size ring buffer."""
#     n = values.shape[0]
#     result = np.empty(n, dtype=np.float32)

#     buffers = np.empty((n_symbols, window), dtype=np.float64)
#     counts = np.zeros(n_symbols, dtype=np.int32)
#     heads = np.zeros(n_symbols, dtype=np.int32)

#     for i in range(n):
#         code = codes[i]
#         val = values[i]

#         count = counts[code]
#         head = heads[code]

#         if count < window:
#             insert_idx = (head + count) % window
#             buffers[code, insert_idx] = val
#             count += 1
#             counts[code] = count
#         else:
#             buffers[code, head] = val
#             head = (head + 1) % window
#             heads[code] = head

#         if np.isnan(val):
#             result[i] = 0.0
#             continue

#         current_head = heads[code]
#         current_count = counts[code]
#         total = 0

#         for j in range(current_count):
#             idx = (current_head + j) % window
#             other = buffers[code, idx]
#             if not np.isnan(other) and other <= val:
#                 total += 1

#         result[i] = np.float32(total / current_count)

#     return result


# def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
#     window = int(window)
#     if window <= 0:
#         raise ValueError("window must be a positive integer")

#     df = pd.read_parquet(input_path, columns=["symbol", "Close"])
#     if df.empty:
#         return np.empty((0, 1), dtype=np.float32)

#     codes, uniques = pd.factorize(df["symbol"], sort=False)
#     n_symbols = int(uniques.size)

#     codes = np.ascontiguousarray(codes.astype(np.int32, copy=False))
#     values = np.ascontiguousarray(df["Close"].to_numpy(dtype=np.float64, copy=False))

#     result = _rolling_rank_kernel(codes, values, window, n_symbols)
#     return result.reshape(-1, 1)  # must be [N, 1]