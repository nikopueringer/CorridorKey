FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Runtime dependencies for OpenCV/video I/O.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy project source.
COPY . .

# Enable OpenEXR support in OpenCV.
ENV OPENCV_IO_ENABLE_OPENEXR=1

ENTRYPOINT ["uv", "run", "python", "corridorkey_cli.py"]
CMD ["--action", "list"]
