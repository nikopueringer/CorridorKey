# Preparing Clips

CorridorKey processes clips that are in the `READY` state. A clip reaches `READY` when it has both input frames and a matching set of alpha hint frames on disk. You can provide these as pre-extracted image sequences or as video files.

## Option 1: Video Files (Recommended)

Drop a video file and its corresponding alpha matte video into a folder. CorridorKey extracts the frames automatically before running inference.

```text
my_shot/
    my_shot.mp4
    my_shot_alpha.mp4
```

Or use `_matte` as the suffix:

```text
my_shot/
    my_shot.mp4
    my_shot_matte.mp4
```

Or name the alpha video `AlphaHint.mp4`:

```text
my_shot/
    my_shot.mp4
    AlphaHint.mp4
```

The wizard detects the clip as `EXTRACTING` and extracts both videos to `Frames/` and `AlphaHint/` before running inference. Extraction is resumable — if interrupted, it picks up from where it left off.

If you only have the input video (no alpha), the clip will be `RAW` after extraction and will need an alpha generator package to proceed.

## Option 2: Image Sequences

Provide pre-extracted frame sequences directly:

```text
my_shot/
    Frames/
        frame_0001.png
        frame_0002.png
        ...
    AlphaHint/
        frame_0001.png
        frame_0002.png
        ...
```

`Frames/` holds the green screen input frames. `AlphaHint/` holds the corresponding alpha matte frames, one per input frame. The frame count in both folders must match exactly.

## Multiple Clips

Put each clip in its own subfolder and point the wizard at the parent:

```text
session/
    actor_jump/
        actor_jump.mp4
        actor_jump_alpha.mp4
    product_spin/
        Frames/
            frame_0001.png
            ...
        AlphaHint/
            frame_0001.png
            ...
```

```shell
corridorkey wizard /path/to/session
```

Video and sequence clips can be mixed in the same session.

## Alpha Hint Format

The alpha hint is a greyscale video or image sequence where white is fully opaque and black is fully transparent. PNG is the recommended format for sequences. Any standard video codec works for video alpha.

The alpha hint does not need to be a perfect matte. A rough garbage matte or a simple threshold mask is sufficient as a hint to guide the model.

## Supported Input Formats

| Type | Formats |
|---|---|
| Video | MP4, MOV, MKV, AVI, and any format FFmpeg can decode |
| Image sequence | PNG, JPEG, TIFF, OpenEXR |

All frames in a sequence clip must use the same format and resolution.

If your input frames are linear light (e.g. EXR from a camera or renderer), enable `input_is_linear` in the wizard settings.

## Frame Naming (Sequences)

Frames must be numbered sequentially. Any consistent zero-padded naming works:

- `frame_0001.png`, `frame_0002.png` ...
- `shot_001.exr`, `shot_002.exr` ...
- `0001.png`, `0002.png` ...

The names in `Frames/` and `AlphaHint/` do not need to match each other. The pipeline matches frames by sort order, not by name.

## Checking Clip State

Run `corridorkey scan` to verify your clips before processing:

```shell
corridorkey scan /path/to/session
```

| State | Meaning |
|---|---|
| EXTRACTING | Video detected, frames not yet extracted. Will extract automatically. |
| READY | Has frames and alpha hint. Ready for inference. |
| RAW | Has frames but no alpha hint. Needs an alpha generator. |
| COMPLETE | Already processed. |

## Related

- [Processing clips](processing.md)
- [Outputs](outputs.md)
- [Clip state machine](../dev/packages/corridorkey/clip-state.md)
