# Preparing Clips

CorridorKey processes clips that are in the `READY` state. A clip reaches `READY` when it has both input frames and a matching set of alpha hint frames on disk. You can provide these as pre-extracted image sequences or as video files — and if your footage is not yet organised into the expected structure, the wizard will offer to do it for you.

## Folder Naming

Folder names (`Input`, `AlphaHint`, `Frames`) are matched case-insensitively. `input/`, `INPUT/`, and `Input/` all work equally well on any operating system.

## Option 1: Drop Anything - Let the Wizard Organise

If your footage is not yet structured, just point the wizard at the folder. At the start of each session it scans for unorganised content and offers to restructure it.

**Loose video files** sitting directly in the clips directory:

```text
session/
    actor_jump.mp4
    product_spin.mp4
```

The wizard detects these and offers to wrap each one into its own clip folder:

```text
session/
    actor_jump/
        Input.mp4
        AlphaHint/          (empty — add your alpha here)
        VideoMamaMaskHint/  (empty)
    product_spin/
        Input.mp4
        AlphaHint/
        VideoMamaMaskHint/
```

**Folders with raw content** (video or image sequence at the top level, no `Input/` subfolder):

```text
session/
    actor_jump/
        random_name.mp4     (any filename)
```

The wizard renames the video to `Input.mp4` in place and creates the hint folders.

**Folders with a flat image sequence**:

```text
session/
    actor_jump/
        frame_0001.png
        frame_0002.png
        ...
```

The wizard moves all images into `Input/`.

After organising, add your alpha hint to `AlphaHint/` and re-scan.

## Option 2: Video Files with Alpha

Drop a video file and its corresponding alpha matte video into a folder. CorridorKey extracts the frames automatically before running inference.

The alpha video can be named in any of these ways (all case-insensitive):

```text
my_shot/
    Input/
        my_shot.mp4
    AlphaHint/
        output.mp4
```

Or as sibling files in the clip root:

```text
my_shot/
    my_shot.mp4
    my_shot_alpha.mp4
```

```text
my_shot/
    my_shot.mp4
    my_shot_matte.mp4
```

```text
my_shot/
    my_shot.mp4
    AlphaHint.mp4
```

The wizard detects the clip as `EXTRACTING` and extracts both videos to `Frames/` and `AlphaHint/` before running inference. Extraction is resumable — if interrupted, it picks up from where it left off.

If you only have the input video (no alpha), the clip will be `RAW` after extraction and will need an alpha generator package to proceed.

## Option 3: Image Sequences

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

The legacy `Input/` folder name is also supported and works identically to `Frames/`.

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
