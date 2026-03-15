# Processing Clips (Wizard)

The wizard is the main way to process footage. It guides you through scanning, reviewing, and running inference interactively.

## Starting the Wizard

```shell
corridorkey wizard /path/to/clips
```

If you omit the path, the wizard will prompt for it.

On Windows you can also drag a clips folder onto the `CorridorKey - Drop Clips Here.bat` shortcut on your Desktop.

## Step 1: Organise (if needed)

Before showing the clip table, the wizard scans for unorganised content — loose video files or folders that contain raw footage without the expected `Input/` subfolder. If it finds any, it shows a summary and asks whether to restructure:

```text
2 loose video file(s) found:
  • /path/to/clips/actor_jump.mp4
  • /path/to/clips/product_spin.mp4

These will be restructured into clip folders with Input/ and empty AlphaHint/ subdirectories.
Organise now? [y/n] (n):
```

Choosing `y` moves each video into its own `{stem}/Input.ext` folder and creates empty `AlphaHint/` and `VideoMamaMaskHint/` subdirectories. The wizard then re-scans and shows the updated clip table.

If everything is already structured, this step is skipped silently.

See [Preparing clips](preparing-clips.md) for details on what the organiser handles.

## Step 2: The Clip State Table

The wizard shows a table of every clip it found and its current state:

```text
Clips in /path/to/clips
 Clip          State        Input  Alpha  Error
 actor_jump    EXTRACTING       0      0
 my_shot       READY           48     48
 another_shot  RAW             60      -
```

The state tells you what the pipeline will do with each clip:

| State | Meaning |
|---|---|
| READY | Has input frames and alpha hint. Will be processed when you choose [i]. |
| EXTRACTING | Source video detected. Frames will be extracted automatically before inference. |
| COMPLETE | Already processed. Will be skipped. |
| RAW | Has input frames but no alpha hint. Needs an alpha generator package. |
| MASKED | Has a VideoMaMa mask hint. Needs VideoMaMa to convert it. |
| ERROR | A previous step failed. See the Error column for details. |

READY and EXTRACTING clips are both processed when you choose [i]. RAW and MASKED clips are skipped with a note.

## Step 3: The Action Menu

```text
  [i] Process 1 READY, 1 to extract clip(s)
  [r] Re-scan directory
  [q] Quit
```

Choose `i` to run inference (extracting any EXTRACTING clips first), `r` to re-scan after adding or changing clips, or `q` to quit.

## Step 4: Inference Settings

Before processing starts, the wizard shows the current settings and asks whether to use them:

```text
 Setting           Value
 input_is_linear   False
 despill_strength  1.0
 auto_despeckle    True
 despeckle_size    400
 refiner_scale     1.0

Use these settings? [y/n] (y):
```

Press Enter to accept the defaults. Enter `n` to adjust each setting interactively.

### What each setting does

`input_is_linear` — Set to True if your frames are linear light (e.g. EXR from a camera or renderer). Leave False for standard sRGB footage (PNG, JPEG from a camera).

`despill_strength` — Controls how aggressively green colour contamination is removed from the subject. 1.0 is maximum. Reduce if skin tones look too magenta.

`auto_despeckle` — Removes small isolated holes and islands in the matte automatically. Leave enabled for most footage.

`despeckle_size` — The maximum size (in pixels) of a matte artefact to remove. Increase if small holes remain; decrease if fine detail like hair is being removed.

`refiner_scale` — Controls the edge refinement pass. 1.0 is the default. Reduce toward 0.0 to speed up processing at the cost of edge quality.

## Model Loading

The first time you run inference in a session, the wizard loads the model and compiles GPU kernels. This takes about one minute and shows a spinner:

```text
Loading model (first run compiles kernels, ~1 min)...
```

Subsequent clips in the same session process immediately because the model stays loaded.

## Extraction

EXTRACTING clips are unpacked before inference runs. The wizard shows progress for each:

```text
Extracting actor_jump  (0 frames)
```

After extraction the clip advances to READY (if an alpha video was also extracted) or RAW (if no alpha was found). RAW clips after extraction are skipped unless an alpha generator is installed.

## Progress

Each clip shows a progress bar while processing:

```text
Processing my_shot  (48 frames)
  Done: 48/48 frames  18.3s  (2.62 fps)
```

After all clips finish, the wizard returns to the main loop so you can re-scan or process more clips.

## Batch Processing (Non-Interactive)

For scripted or automated use, `corridorkey process` runs without any prompts:

```shell
corridorkey process /path/to/clips --despill 0.9 --fg-format exr
```

See [Processing commands](../dev/packages/corridorkey-cli/processing-commands.md) for all available flags.

## Related

- [Outputs](outputs.md)
- [Preparing clips](preparing-clips.md)
- [Troubleshooting](troubleshooting.md)
- [Processing commands](../dev/packages/corridorkey-cli/processing-commands.md)
