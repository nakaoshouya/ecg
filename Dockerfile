# ベースイメージ: CUDA 12.1 + cuDNN 8 + Ubuntu 22.04
# PyTorch 2.5.1+cu121 に合わせたバージョン
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# 環境変数設定
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    MPLBACKEND=Agg

# システムパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3-setuptools \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# python3 -> python3.11 のシンボリックリンク
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# pipのアップグレード
RUN pip3 install --upgrade pip

# 作業ディレクトリ設定
WORKDIR /app

# ── 依存パッケージのインストール ──────────────────────────────
# requirements.txt を先にコピーすることでキャッシュを最大限活用する
# （コードを変更してもpipのレイヤーは再実行されない）
COPY requirements.txt .

# PyTorchはCUDA版を--index-urlで明示指定（requirements.txtには書かない）
RUN pip3 install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchaudio==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# その他の依存パッケージ
RUN pip3 install --no-cache-dir -r requirements.txt

# ── プロジェクトファイルのコピー ──────────────────────────────
COPY available_gpu.py .
COPY main.py          .
COPY train.py         .
COPY trained_model.pth .
COPY src/             ./src/
COPY data/            ./data/

# ── インタラクティブ実行用の設定 ──────────────────────────────
# デフォルトはbashシェル起動。コンテナに入って自分でスクリプトを実行する
#   python3 main.py
#   python3 train.py
#   python3 available_gpu.py
CMD ["/bin/bash"]
