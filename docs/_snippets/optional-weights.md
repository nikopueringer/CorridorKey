!!! tip "Optional — GVM and VideoMaMa weights"
    These modules generate Alpha Hints automatically but have large model files
    and extreme hardware requirements. Installing them is **completely optional**;
    you can always provide your own Alpha Hints from other software.

    **GVM** (~80 GB VRAM required):

    ```bash
    uv run hf download geyongtao/gvm --local-dir gvm_core/weights
    ```

    **VideoMaMa** (originally 80 GB+ VRAM; community optimisations bring it
    under 24 GB, though not yet fully integrated here):

    ```bash
    # Fine-tuned VideoMaMa weights
    uv run hf download SammyLim/VideoMaMa \
      --local-dir VideoMaMaInferenceModule/checkpoints/VideoMaMa

    # Stable Video Diffusion base model (VAE + image encoder, ~2.5 GB)
    # Accept the licence at stabilityai/stable-video-diffusion-img2vid-xt first
    uv run hf download stabilityai/stable-video-diffusion-img2vid-xt \
      --local-dir VideoMaMaInferenceModule/checkpoints/stable-video-diffusion-img2vid-xt \
      --include "feature_extractor/*" "image_encoder/*" "vae/*" "model_index.json"
    ```
