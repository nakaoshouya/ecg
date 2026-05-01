import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def extract_1d_signal_from_clean_mask(clean_mask):
    clean_mask = cv2.cvtColor(clean_mask, cv2.COLOR_BGR2GRAY)
    H, W = clean_mask.shape
    ys, xs = np.where(clean_mask > 0)
    # ★ 波形が1ピクセルも無い → ゼロ波形で返す（SNR的に最悪回避）
    if len(xs) == 0:
        return np.zeros(W, dtype=np.float32), (0, 0, W - 1, H - 1), W

    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()
    x_size = xmax - xmin

    crop = clean_mask[ymin:ymax + 1, xmin:xmax + 1]
    Hc, Wc = crop.shape

    # 各 x 列で白ピクセルの y 位置を1つ決める
    y_curve = np.full(Wc, np.nan, dtype=np.float32)

    all_ys = np.where(crop > 0)[0]
    if all_ys.size == 0:
        return np.zeros(Wc, dtype=np.float32), (xmin, ymin, xmax, ymax), x_size
    
    global_median_y = np.median(all_ys)

    gap_thresh = 100  # ここが「npx以上離れたら別波形」閾値
    prev_y = 0    # 直前xで採用したy（追跡用）
    for x in range(Wc):
        col_ys = np.where(crop[:, x] > 0)[0]
        if col_ys.size == 0:
            continue
    
        col_ys = np.sort(col_ys)
    
        # --- 1) gap_thresh でブロック分割 ---
        # split_idx: 分割位置（diff>gap_thresh の直後で切る）
        diffs = np.diff(col_ys)
        split_idx = np.where(diffs > gap_thresh)[0] + 1
        blocks = np.split(col_ys, split_idx)
    
        # --- 2) ブロックごとの代表y（中央値） ---
        reps = np.array([np.median(b) for b in blocks], dtype=np.float32)
    
        # --- 3) どのブロックを採用するか決める ---
        if prev_y is None:
            # 最初は「一番太いブロック」優先（=白画素数最大）
            # ※ ここは global_median_y に近いブロックを選ぶ、でもOK
            sizes = np.array([len(b) for b in blocks])
            best = int(np.argmax(sizes))
        else:
            # 前のyに最も近いブロック
            best = int(np.argmin(np.abs(reps - prev_y)))
    
        chosen = blocks[best]
    
        # --- 4) chosenブロックから y を1点に集約（あなたの方針を踏襲） ---
        local_median = np.median(chosen)
        if local_median < global_median_y:
            y_curve[x] = np.percentile(chosen, 5)
        else:
            y_curve[x] = np.percentile(chosen, 95)
    
        # --- 5) 次の列のために prev_y を更新 ---
        prev_y = y_curve[x]


    # 有効列（NaNじゃない列）を確認
    valid = ~np.isnan(y_curve)

    # ★ 点が2つ以上ない場合→補間できないのでゼロで返す
    if valid.sum() < 2:
        return np.zeros(Wc, dtype=np.float32), (xmin, ymin, xmax, ymax)

    # 線形補間で NaN を埋める
    x_valid = np.where(valid)[0].astype(np.float32)
    y_valid = y_curve[valid].astype(np.float32)
    y_curve = np.interp(np.arange(Wc, dtype=np.float32), x_valid, y_valid).astype(np.float32)

    # 座標系を「上が正」に変換し、ベースライン0にシフト
    y_curve = (Hc - 1) - y_curve
    baseline = np.median(y_curve).astype(np.float32)
    signal = y_curve - baseline

    return pd.DataFrame(signal.astype(np.float32)), ymin, ymax, xmin, xmax

def df2csv(df, csv_path):
    csv_path = Path(csv_path)
    df.to_csv(csv_path, index=False)
    print("CSV完了")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_graph(csv_path, ymin, ymax, xmin, xmax):

    # --- スケール計算（維持） ---
    y_range = ymax - ymin
    x_range = xmax - xmin

    px_per_block = x_range / 50
    s_per_px = 0.2 / px_per_block
    mV_per_px = 0.5 / px_per_block

    # --- データ読み込み ---
    df = pd.read_csv(csv_path)
    signal = df.iloc[:, 0].values

    x = np.arange(len(signal)) * s_per_px
    y = signal * mV_per_px

    # --- 描画 ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y, color="black", linewidth=1)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("mV")
    ax.set_title("ECG Digital Data Graph")

    # --- y軸固定 ---
    ax.set_ylim(-3.0, 3.0)

    # --- ★ここが重要：0.5刻みで数字表示 ---
    ax.set_xticks(np.arange(0, max(x) + 0.5, 0.5))
    ax.set_yticks(np.arange(-3.0, 3.0 + 0.5, 0.5))

    # --- グリッド（ECG仕様） ---

    # 小マス（細かい補助線）
    ax.set_xticks(np.arange(0, max(x), 0.1), minor=True)
    ax.set_yticks(np.arange(-3.0, 3.0, 0.1), minor=True)

    # グリッド表示
    ax.grid(which='major', color='red', linewidth=0.8)
    ax.grid(which='minor', color='red', linewidth=0.3, alpha=0.3)

    # --- 枠線消す（ECG風） ---
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_aspect(0.2 / 0.5)

    plt.tight_layout()
    plt.show()
