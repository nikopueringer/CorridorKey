# Troubleshooting

Run `corridorkey doctor` first. It checks the most common failure points and tells you exactly what is wrong.

## Inference model not found

```text
FAIL  inference model  not found in ~/.config/corridorkey/models
```

The model was not downloaded. Run `corridorkey init` and choose to download when prompted. If the download fails, see the manual download instructions printed by init.

## FFmpeg not found

```text
FAIL  ffmpeg  not found - install FFmpeg
```

FFmpeg is not installed or not on PATH.

On Windows, download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin/` folder to your PATH environment variable.

On macOS with Homebrew:

```shell
brew install ffmpeg
```

On Ubuntu/Debian:

```shell
sudo apt install ffmpeg
```

After installing, run `corridorkey doctor` to confirm.

## corridorkey: command not found

The tool installed but the install location is not on PATH. Close and reopen your terminal. If that does not help, find where `uv` installed the tool:

```shell
uv tool dir
```

Add the path printed there to your PATH environment variable.

## Clips not detected

If the wizard shows no clips or fewer clips than expected, the most likely cause is that the folder structure is not recognised.

The wizard scans for `Input/`, `Frames/`, or `Source/` subdirectories (case-insensitive). If your footage is in a flat folder without these subdirectories, the wizard will offer to organise it at the start of the session. Choose `y` when prompted.

If you declined the organise prompt and clips are still missing, check that each clip folder contains at least one of:

- `Input/` or `input/` — image sequence or video inside
- `Frames/` or `frames/` — image sequence
- `Source/` — video file
- A video file named `Input.mp4` (or any case/extension) directly in the clip root

## Clips stuck in RAW state

```text
 my_shot  RAW  60  -
```

RAW means the clip has input frames but no alpha hint. CorridorKey needs an alpha generator package to produce the alpha hint before inference can run. Install one and re-scan.

If you expected the clip to be READY, check that the `AlphaHint/` folder exists inside the clip directory and contains the same number of frames as `Frames/` or `Input/`.

## Clips stuck in EXTRACTING state

```text
 my_shot  EXTRACTING  0  0
```

EXTRACTING means a source video was detected but frames have not been extracted yet. This happens automatically when you choose [i] in the wizard. If the clip stays in EXTRACTING after processing:

- Check that FFmpeg is installed (`corridorkey doctor`).
- Check that the source video is not corrupted (try opening it in a media player).
- Check the error column in the wizard table for a specific message.

## CUDA out of memory

```text
RuntimeError: CUDA out of memory
```

The GPU does not have enough VRAM for the current settings. Try:

1. Close other applications using the GPU.
2. Reduce `refiner_scale` to 0.5 or 0.0 in the wizard settings.
3. Process one clip at a time rather than a large batch.

If the error persists on a 4 GB GPU, the model may not fit. CPU processing is the fallback:

```shell
corridorkey process /path/to/clips --device cpu
```

## First frame takes very long (kernel compilation)

On the first run after installation, PyTorch compiles GPU kernels for your specific hardware. This takes approximately one minute and is normal. The wizard shows a spinner during this step. Subsequent runs use the cached compiled kernels and start immediately.

## Output frames are too dark or too bright

If your input footage is linear light (EXR from a camera or renderer) and you did not enable `input_is_linear`, the pipeline will treat it as sRGB and the output will look wrong. Re-run with `input_is_linear` enabled in the wizard settings.

## Matte has holes or speckles

Small holes in the matte are removed by `auto_despeckle`. If holes remain, increase `despeckle_size` in the wizard settings. If fine detail like hair is being removed, decrease `despeckle_size`.

For persistent edge problems, try reducing `despill_strength` slightly (e.g. 0.8) as aggressive despill can affect semi-transparent areas.

## Config file not found warning

```text
WARN  config file  not found - run `corridorkey init` to create it
```

This is a warning, not a failure. CorridorKey uses built-in defaults when no config file is present. Run `corridorkey init` or `corridorkey config init` to create the file.

## Related

- [First run](first-run.md)
- [Setup commands](../dev/packages/corridorkey-cli/setup-commands.md)
- [Configuration](../dev/packages/corridorkey/configuration.md)
