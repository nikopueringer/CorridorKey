# CorridorKey 2 for Nuke: Local / SSH

Experimental Python Group for Linux Nuke 15.1. It bakes a frame range with the
separately installed [public CK2 runtime](https://github.com/nikopueringer/CorridorKey2.0),
then reads the resulting RGBA EXRs into the graph. There are two execution modes:

- **Local:** use a CUDA GPU on the Nuke workstation.
- **SSH:** upload the exported frames to a GPU workstation, execute there, and
  download verified results. Shared storage is not required.

This adapter does not contain the commercial OFX plugin, model code or weights.
It does not require a commercial plugin activation key. The separately installed
runtime/model remain subject to their [CorridorKey licence](https://github.com/nikopueringer/CorridorKey2.0/blob/master/LICENSE),
including the additional terms for redistribution, paid inference services,
commercial software integration and large organisations. This is an independent
experimental integration, not an official Corridor Digital plugin.

## Install for a session

Copy this `nuke` directory somewhere readable by the workstation. In Nuke's
Script Editor, execute the following with the actual absolute directory:

```python
import nuke

nuke.pluginAddPath("/absolute/path/to/CorridorKey/nuke")
import ck2_nuke.ui

node = ck2_nuke.ui.create_node()
```

For a persistent menu entry, add only the `nuke.pluginAddPath(...)` line to your
existing Nuke startup configuration. `menu.py` adds **CorridorKey > CK2 Local or
SSH** to the Nodes menu. No installer edits startup files automatically.

The Nuke-side code uses Python's standard library and Nuke/PySide. Do not install
PyTorch into Nuke. The job client needs a separate Python 3.9+ interpreter.
Nuke 16/PySide6 has an import fallback but has not been qualified.

## Runtime setup on each GPU machine

Install the public CK2 repository and its dependencies/weights using its own
instructions. In that same external environment, install the EXR validation tool:

```sh
/absolute/path/to/ck2/venv/bin/python -m pip install OpenEXR
```

CK2 currently requires Linux with an NVIDIA CUDA GPU, supported drivers and
`ffmpeg`. This adapter does not add a CPU, Metal or ROCm inference backend.
The initial integration targets CK2 commit `485f5e9ab759ead6138cc990776184b59d6ea971`.

On the node's **Setup** tab, set **Local CUDA Python** to that environment's
absolute Python executable and **Local CK2 runtime** to the repository directory.
For local mode, **Client Python** can be empty; the CUDA Python is used.

For SSH mode, set:

- **SSH alias:** a hostname or existing OpenSSH alias, e.g. `gpu-box`.
- **Remote CUDA Python / Remote CK2 runtime:** absolute paths on the GPU machine.
- **Remote job folder:** an absolute writable directory on the GPU machine.
- **Client Python:** a local Python 3.9+ executable; defaults to `/usr/bin/python3`.
- **Local job folder:** a writable local directory for exported frames/results.

SSH must already work without prompts (`ssh -o BatchMode=yes gpu-box true`).
Host keys are checked by normal OpenSSH policy. Nothing opens a network port or
changes SSH configuration. Credentials are never copied into job bundles.

Defaults can also be set before Nuke starts using `CK2_JOBS`, `CK2_COLORSPACE`,
`CK2_CLIENT_PYTHON`, `CK2_PYTHON`, `CK2_RUNTIME`, `CK2_REMOTE_HOST`,
`CK2_REMOTE_PYTHON`, `CK2_REMOTE_RUNTIME` and `CK2_REMOTE_JOBS`.

## Work with a shot

1. Connect the plate to input 0. Optionally connect an alpha matte to input 1.
2. Choose **Local** or **SSH**. Range defaults to **Source**; choose **Custom** to
   enter first/last frames explicitly. Frames are contiguous, with no frame step.
3. Choose **None**, **First frame**, or **All frames** for the alpha hint. First
   frame means the first frame of the submitted range. The alpha channel is
   exported into grayscale; hint RGB is ignored. Hint format/PAR must match.
4. Start with **Half** internal resolution and a short range. CK2 preserves the
   full exported output dimensions. GPU memory selection is automatic unless
   **Low GPU memory** is enabled. Despill defaults to zero.
5. Click **Render range**. Export uses Nuke's normal progress/cancel dialog;
   inference/transport runs outside the Nuke process. GUI status is polled once
   per second. **Cancel** requests that the worker terminate its own inference
   process group; it never terminates other Nuke/GPU jobs.
6. A successfully verified result is loaded and **Use baked result** is enabled.
   Disable it to see the upstream plate. **Load completed result** can reload a
   completed job after reopening the script.

This is a disk bake, not an interactive per-pixel keyer. Upstream graph edits do
not invalidate it automatically. Render again after upstream edits. Local changes
to processing controls turn off **Use baked result**. Jobs and existing outputs
are retained; every new render uses a new UUID directory.

## Colour and geometry contract

**Linear Rec.709 colour space** must be the name in your Nuke/OCIO configuration
that represents linear light with Rec.709/sRGB primaries. In the default Nuke
configuration this is `linear`. In ACES 1.x it is commonly
`Utility - Linear - sRGB`. Select the correct name for your config; an arbitrary
ACES working space or log space is not a substitute.

Nuke converts from the working space to that space when exporting 32-bit float
RGB EXRs. CK2 receives `--frame_dir_linear`, writes float EXRs, and does not use
lossy video intermediates. Results are **straight linear Rec.709 RGBA**; a Read
node converts them back to the working space, followed by Premult. The adapter
restores original frame numbering and pixel aspect, verifies dimensions, rejects
non-finite values/invalid alpha, and verifies SHA-256 checksums on both ends.
HDR and negative RGB are preserved by transport; the learned model itself may
change those pixels. Synthetic transport preservation is not optical/key quality.

The full format rectangle `[0, width] x [0, height]` is exported. Overscan outside
that rectangle and multiview are not supported in this initial version. Frames
must keep a constant format and pixel aspect across the range.

Inference starts with Python/NumPy/PyTorch seed 0, `PYTHONHASHSEED=0`, deterministic
PyTorch algorithms and a fixed cuBLAS workspace policy. Unsupported deterministic
kernels fail with an error rather than silently changing this policy.

**Known qualification limit:** the initial three-frame test on the same RTX 4080
produced repeatable results within each launch mode, but local-from-Nuke and SSH
inference differed by up to about 0.028 in alpha despite identical exported and
prepared input checksums. Fixing RNG/hash seeds did not remove the difference.
The cause remains unresolved. Transport is verified; numerical parity between
execution modes is not qualified. Review mattes before using this experimental
adapter in production. Equivalence across different GPUs/drivers is also untested.

For Half/Quarter, this adapter downsamples exported linear RGB/hints with an area
filter before inference and upsamples the straight RGBA result with Lanczos4,
clamping only alpha to [0,1]. CK2 itself runs at `--model_scale 1.0`: the targeted
upstream revision accepts other values but fails when writing its optimized EXR
output. Reduced resolution can soften detail; it is a preview/iteration option.

## Logs, cancellation and recovery

Each job contains `job.json` (settings/input hashes), `status.json`, `client.log`,
`transport.log`, `inference.log`, `result.json` (output hashes, validation stats,
runtime provenance), and `output/rgba.%06d.exr`. Raw CK2 output and scratch files
remain on the execution machine. There is no automatic cleanup or service daemon.
The inference timeout is two hours per job by default. A lost SSH connection may
leave the worker running until its timeout; no successful result is claimed until
download and verification finish.

For a completed remote job whose download was interrupted, retry retrieval
without running inference again:

```sh
python3 /absolute/path/to/nuke/ck2_nuke/client.py /local/jobs/JOB_ID \
  --host gpu-box --remote-root /remote/jobs \
  --python /remote/ck2/venv/bin/python --runtime /remote/ck2 --retrieve
```

The terminal client runs in the foreground. For cancellation, create an empty
`CANCEL` file in its local job directory. Remote cancellation requires the SSH
connection to work. The worker also honours `CANCEL` in its remote job directory.

## Validation

Dependency-light tests use synthetic EXRs and a fake inference CLI to verify
process/transport contracts. They do not download or run the model:

```sh
python3 -m pip install -r nuke/tests/requirements.txt
python3 -m pytest -c /dev/null -p no:cacheprovider nuke/tests -q
ruff check nuke
ruff format --check nuke
```

`tests/native_smoke.py` runs inside Nuke 15.1 with `CK2_TEST_ROOT` set to a fresh
test directory. It checks native Group creation, 32-bit export, negative/HDR RGB,
PAR=2 and alpha-hint extraction. Set `CK2_TEST_RUNTIME` and `CK2_TEST_PYTHON` to
add a bounded three-frame local GPU bake and a native downstream EXR render.
`CK2_TEST_SOURCE` can point to an existing 1001–1003 EXR sequence for that bake.
No user script or startup file is edited.
