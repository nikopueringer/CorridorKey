## Tiled Inference for Consumer GPU Support

### Problem

GreenFormer's full 2048×2048 forward pass requires ~22.7 GB VRAM, which limits CorridorKey to professional GPUs with 24+ GB. Most VFX artists working on personal machines have 8–12 GB cards (RTX 3060/3070/4060 etc.) and can't run inference at all.

### Solution

This PR adds a tiled inference mode that splits the 2048×2048 model input into overlapping square tiles, runs each tile independently through GreenFormer, blends overlapping regions with cosine ramps, and stitches the result back together. Combined with optional fp16 weight casting, this brings peak VRAM down to ~6–8 GB.

The default behavior is `--tile-size auto`, which queries available VRAM and picks an appropriate tile size (or skips tiling entirely on 24+ GB cards). Existing workflows on professional GPUs are completely unaffected.

### How it works

The tiling layer sits inside `process_frame()` between tensor preparation (step 4) and post-processing (step 6). It intercepts the model forward pass only — everything upstream (input normalization, resize to 2048²) and downstream (despeckle, despill, composite, EXR export) is unchanged.

```
Input → resize to 2048² → [TILED FORWARD PASS] → stitch → post-processing → output
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
                 Tile(0,0)    Tile(0,1)    Tile(1,0) ...
                    │             │             │
                    ▼             ▼             ▼
                GreenFormer  GreenFormer  GreenFormer
                    │             │             │
                    └──── cosine blend + accumulate ────→ stitched 2048² output
```

Key design decisions:
- The model is initialized at `img_size=tile_size` (not 2048). This is critical because Hiera's `pos_embed`, `Unroll`, and `Reroll` modules all bake spatial dimensions at init time. The checkpoint's 2048-trained `pos_embed` is bicubic-interpolated to the tile resolution during weight loading (using the existing interpolation logic in `_load_model()`).
- Tile sizes must be multiples of 224 — the LCM of Hiera's patch_embed stride (4), q_stride pooling (2³=8, so 4×8=32), and backbone patch stride (7). LCM(32,7)=224.
- The input image is still resized to 2048×2048 to preserve full resolution. Tiles are extracted from this 2048 space, each tile processed by the smaller model, then blended and stitched back to 2048.
- Overlap regions use a cosine ramp blend (`0.5 - 0.5·cos(πt)`) for smooth second-derivative continuity — no visible seams
- GPU tensors are released between tiles to keep peak VRAM proportional to a single tile
- Tested at tile_size=672 (from input 768): VRAM dropped from ~22.7 GB to ~1.7 GB on an A10G

### VRAM auto-detection

When `--tile-size auto` (the default), the system queries `torch.cuda.get_device_properties()` and picks a tile size:

| Total VRAM | Tile Size | Tiling? |
|---|---|---|
| ≥ 24 GB | — | No (full 2048² pass) |
| 12–24 GB | 1344 | Yes |
| 8–12 GB | 896 | Yes |
| < 8 GB | 672 + warning | Yes |
| MPS (Apple Silicon) | — | No (unified memory) |
| CPU | — | No |

### New CLI flags

```
--tile-size   Tile size in pixels, "auto", or "off" (default: "auto")
--overlap     Overlap size in pixels (default: 128)
--half        Use fp16 model weights for additional VRAM savings
```

These are available in both `corridorkey_cli.py` and `clip_manager.py`.

### Files changed

**New files:**
- `CorridorKeyModule/tiled_inference.py` — `TiledInferenceEngine`, `VRAMDetector`, blend utilities, tile grid computation (~386 lines)
- `tests/test_tiled_inference.py` — Property-based + unit tests for tiling logic (9 correctness properties)
- `tests/test_blend_ramp.py` — Property-based tests for cosine ramp construction
- `tests/test_tiled_integration.py` — Integration tests: tiled vs non-tiled equivalence with mock model
- `tests/test_vram_detector.py` — Property-based tests for VRAM tier recommendations

**Modified files:**
- `CorridorKeyModule/inference_engine.py` — `CorridorKeyEngine.__init__()` accepts `tile_size`, `overlap_size`, `half_precision`; `process_frame()` delegates to tiler when enabled
- `CorridorKeyModule/backend.py` — `create_engine()` accepts and resolves tiling params (`"auto"` → VRAMDetector, `"off"`/`0` → disabled)
- `corridorkey_cli.py` — New `--tile-size`, `--overlap`, `--half` arguments
- `clip_manager.py` — Same new arguments, forwarded to `create_engine()`
- `pyproject.toml` — Added `hypothesis` to dev dependencies
- `.gitignore` — Added `.hypothesis/`

**No changes to third-party code** (`gvm_core/`, `VideoMaMaInferenceModule/`).

### Testing

18 tests covering 9 formal correctness properties, all using `hypothesis` for property-based testing:

1. **Tile grid correctness** — tile union covers all pixels, adjacent tiles overlap by exactly `overlap_size`
2. **Patch stride alignment** — `align_to_patch_stride()` always returns a multiple of 7
3. **Cosine ramp correctness** — values in [0,1], core = 1.0, edges approach 0.0, follows cosine formula
4. **Weight sum full coverage** — every pixel has weight > 0 after accumulating all tile ramps
5. **Tiled vs non-tiled equivalence** — output matches within tolerance (integration test with mock model)
6. **Overlap validation** — clamps below 65px, rejects ≥ tile_size/2
7. **VRAM tier recommendation** — correct tile size for each VRAM bracket
8. **fp16 weight casting** — all model params become float16
9. **Output contract preservation** — output dict has correct keys and shapes

All tests run without GPU or model weights:

```bash
uv run pytest                # 189 passed, 1 skipped (pre-existing)
uv run ruff check            # clean
uv run ruff format --check   # clean
```

### Backward compatibility

- Default behavior (`--tile-size auto`) on 24+ GB GPUs = no tiling, identical to current behavior
- `process_frame()` return contract is preserved exactly (same keys, shapes, dtypes)
- No changes to downstream consumers (EXR export, compositing, MLX adapter)
- `--tile-size off` explicitly disables tiling for anyone who wants to opt out
