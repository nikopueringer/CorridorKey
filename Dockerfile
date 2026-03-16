FROM ghcr.io/astral-sh/uv:0.7-python3.11-bookworm-slim

WORKDIR /app

# Create non-root user upfront.
RUN useradd --create-home --uid 1000 appuser

# Runtime dependencies for OpenCV/video I/O.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching.
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy project source.
COPY --chown=appuser:appuser . .

# Install the project itself (cheap, just sets up editable/entry points).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

USER appuser

# Enable OpenEXR support in OpenCV.
ENV OPENCV_IO_ENABLE_OPENEXR=1

ENTRYPOINT ["/app/.venv/bin/python", "corridorkey_cli.py"]
CMD ["--action", "list"]
