import sys, os
sys.path.append(os.path.dirname(__file__))  # 确保能导入同目录下的 .pyd

import numpy as np
import pandas as pd
import rolling_rank_cpp

def ops_rolling_rank(input_path: str, window: int = 20):
    window = int(window)
    if window <= 0:
        raise ValueError("window must be a positive integer")

    df = pd.read_parquet(input_path, columns=["symbol", "Close"])
    if df.empty:
        return np.empty((0, 1), dtype=np.float32)

    df["symbol"] = df["symbol"].astype("category")
    codes  = df["symbol"].cat.codes.to_numpy(np.int32,   copy=False)
    values = df["Close"].to_numpy(          np.float32,  copy=False)  # ✅ float32 对齐 C++
    # 可确保 C 连续：
    # codes  = np.ascontiguousarray(codes,  dtype=np.int32)
    # values = np.ascontiguousarray(values, dtype=np.float32)

    n_symbols = len(df["symbol"].cat.categories)

    res = rolling_rank_cpp.rolling_rank(codes, values, window, int(n_symbols))
    return res.reshape(-1, 1).astype(np.float32, copy=False)

