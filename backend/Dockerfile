FROM nvidia/cuda:11.6.2-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        libcudnn8=8.4.1.50-1+cuda11.6 \
        libnvinfer8=8.4.3-1+cuda11.6 \
        libnvinfer-plugin8=8.4.3-1+cuda11.6 \
        libnvonnxparsers8=8.4.3-1+cuda11.6 \
        python3 \
        python3-dev \
        python3-pip \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir -r requirements.txt \
    && apt-get purge --yes --auto-remove build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libcublas-11-6 \
        libcufft-11-6 \
        libcurand-11-6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-db.txt .
RUN python3 -m pip install --no-cache-dir -r requirements-db.txt

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY alembic.ini .
COPY alembic ./alembic
COPY app ./app
COPY benchmark ./benchmark
COPY tests ./tests

ENV LD_LIBRARY_PATH=/usr/local/cuda-11.6/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

EXPOSE 8000

CMD ["sh", "-c", "python3 -m alembic upgrade head && exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port \"${API_PORT:-8000}\""]
