"""Run with the external CK2 Python environment, never with Nuke's Python."""

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ck2_nuke.jobs import digest, read_json, verify_inputs, write_json


def status(directory, state, message=""):
    write_json(Path(directory) / "status.json", {"state": state, "message": message, "time": time.time()})
    print(state + (": " + message if message else ""), flush=True)


def command(directory, runtime, job):
    inputs = Path(directory) / ("prepared" if job["model_scale"] != 1.0 else "")
    return (
        [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("runtime_entry.py")),
            str(Path(runtime) / "infer.py"),
            "--input",
            str(inputs / "source"),
            "--output_dir",
            str(Path(directory) / "raw"),
            "--temp_dir",
            str(Path(directory) / "scratch"),
            "--output_format",
            "exr",
            "--exr_dtype",
            "float",
            "--outputs",
            "cutout",
            "--frame_dir_linear",
            "--cutout_linear",
            "--frame_dir_fps",
            str(job["fps"]),
            "--model_scale",
            "1.0",
            "--despill_strength",
            str(job["despill"]),
            "--hint_quality",
            str(job["hint_quality"]),
        ]
        + (["--low_vram"] if job["low_vram"] else [])
        + {
            "none": [],
            "first": ["--hint_first_frame", str(inputs / "hint/000000.exr")],
            "all": ["--hint_video", str(inputs / "hint"), "--no_carry_hint"],
        }[job["hint"]]
    )


def model_size(job):
    return tuple(max(1, round(job[key] * job["model_scale"])) for key in ("width", "height"))


def prepare(directory, job):
    if job["model_scale"] == 1.0:
        return
    import cv2

    # CK2 485f5e9's --model_scale fails on its optimized EXR output path.
    # Perform explicit linear-light resampling here and keep upstream at 1.0.
    for name in job["inputs"]:
        destination = Path(directory) / "prepared" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(Path(directory) / name), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Cannot decode " + name)
        image = cv2.resize(image, model_size(job), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]):
            raise ValueError("Cannot write prepared input " + name)


def normalize(directory, job):
    # Imported only in the CUDA worker. The upstream EXR writer drops PAR.
    import cv2
    import Imath
    import numpy as np
    import OpenEXR

    directory = Path(directory)
    raw = directory / "raw/source_cutout_exr"
    expected = [raw / ("cutout_%06d.exr" % i) for i in range(job["last"] - job["first"] + 1)]
    if set(raw.glob("*.exr")) != set(expected):
        raise ValueError("Runtime output count or numbering differs from the submitted range")
    output = directory / "output"
    output.mkdir(exist_ok=False)
    files = {}
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    stats = []
    for index, source in enumerate(expected):
        image = OpenEXR.InputFile(str(source))
        try:
            header = image.header()
            window = header["dataWindow"]
            size = (window.max.x - window.min.x + 1, window.max.y - window.min.y + 1)
            if size != model_size(job) or not {"R", "G", "B", "A"} <= set(header["channels"]):
                raise ValueError("Runtime changed the image size or omitted RGBA")
            channels = {c: image.channel(c, pixel_type) for c in "RGBA"}
            arrays = {c: np.frombuffer(data, dtype=np.float32) for c, data in channels.items()}
            if not all(np.isfinite(a).all() for a in arrays.values()):
                raise ValueError("Non-finite runtime output")
            if arrays["A"].min() < 0 or arrays["A"].max() > 1:
                raise ValueError("Runtime alpha outside [0, 1]")
            if size != (job["width"], job["height"]):
                arrays = {
                    c: cv2.resize(
                        a.reshape(size[1], size[0]), (job["width"], job["height"]), interpolation=cv2.INTER_LANCZOS4
                    )
                    for c, a in arrays.items()
                }
                arrays["A"] = arrays["A"].clip(0, 1)
                channels = {c: a.astype(np.float32).tobytes() for c, a in arrays.items()}
                size = (job["width"], job["height"])
            # Fresh header prevents stale ACES/source metadata being copied onto Rec.709 pixels.
            header = OpenEXR.Header(*size)
            header["channels"] = {c: Imath.Channel(pixel_type) for c in "RGBA"}
            header["pixelAspectRatio"] = float(job["pixel_aspect"])
            header["chromaticities"] = Imath.Chromaticities(
                Imath.V2f(0.64, 0.33), Imath.V2f(0.30, 0.60), Imath.V2f(0.15, 0.06), Imath.V2f(0.3127, 0.3290)
            )
            name = "output/rgba.%06d.exr" % (job["first"] + index)
            target = OpenEXR.OutputFile(str(directory / name), header)
            try:
                target.writePixels(channels)
            finally:
                target.close()
            files[name] = digest(directory / name)
            stats.append(
                {
                    "frame": job["first"] + index,
                    "alpha_min": float(arrays["A"].min()),
                    "alpha_max": float(arrays["A"].max()),
                    "rgb_max": max(float(arrays[c].max()) for c in "RGB"),
                }
            )
        finally:
            image.close()
    return files, stats


def validate_headers(directory, job):
    import OpenEXR

    for name in job["inputs"]:
        image = OpenEXR.InputFile(str(Path(directory) / name))
        try:
            header = image.header()
            window = header["dataWindow"]
            size = (window.max.x - window.min.x + 1, window.max.y - window.min.y + 1)
            if size != (job["width"], job["height"]):
                raise ValueError("Input format changes within the range: " + name)
            if abs(float(header.get("pixelAspectRatio", 1.0)) - job["pixel_aspect"]) > 1e-6:
                raise ValueError("Input pixel aspect differs from the source: " + name)
            if not {"R", "G", "B"} <= set(header["channels"]):
                raise ValueError("Missing RGB input channels: " + name)
        finally:
            image.close()


def run(directory, runtime):
    directory, runtime = Path(directory).resolve(), Path(runtime).resolve()
    job = read_json(directory / "job.json")
    process = None
    cancelled = False
    lock = None

    def terminate(_signum, _frame):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        verify_inputs(directory, job)
        if not (runtime / "infer.py").is_file():
            raise ValueError("Missing CK2 infer.py in " + str(runtime))
        if (directory / "result.json").exists() or (directory / "raw").exists():
            raise ValueError("Job already ran; create a new job to preserve existing results")
        # Fail before expensive inference if the output-validation dependency is missing.
        validate_headers(directory, job)

        # Serialize this user's adapter jobs across local/SSH job roots on the host.
        # This lock never changes other applications' GPU usage.
        lock = (Path(tempfile.gettempdir()) / ("ck2-nuke-gpu-%s.lock" % os.getuid())).open("a")
        waiting = time.monotonic()
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if cancelled or (directory / "CANCEL").exists():
                    raise InterruptedError("Cancelled while waiting for another CK2 job") from None
                if time.monotonic() - waiting > job["timeout"]:
                    raise TimeoutError("Timed out waiting for another CK2 job") from None
                status(directory, "queued", "Another CK2 adapter job is using this machine")
                time.sleep(1)

        argv = command(directory, runtime, job)
        commit = subprocess.run(
            ["git", "-C", str(runtime), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        execution = {
            "command": argv,
            "runtime": str(runtime),
            "infer_sha256": digest(runtime / "infer.py"),
            "runtime_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "seed": 0,
            "python_hash_seed": 0,
            "deterministic_algorithms": True,
        }
        write_json(directory / "execution.json", execution)
        status(directory, "running", "CUDA inference")
        started = time.monotonic()
        environment = os.environ.copy()
        environment["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        environment["PYTHONHASHSEED"] = "0"
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        prepare(directory, job)
        with (directory / "inference.log").open("wb") as log:
            process = subprocess.Popen(
                argv, cwd=str(runtime), env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            while process.poll() is None:
                if cancelled or (directory / "CANCEL").exists():
                    raise InterruptedError("Cancelled by user")
                if time.monotonic() - started > job["timeout"]:
                    raise TimeoutError("Inference exceeded the job timeout")
                time.sleep(0.25)
            if process.returncode:
                raise RuntimeError("CK2 exited %s; see inference.log" % process.returncode)
        if cancelled or (directory / "CANCEL").exists():
            raise InterruptedError("Cancelled by user")
        status(directory, "validating", "Checking RGBA, frame range, HDR and pixel aspect")
        files, stats = normalize(directory, job)
        write_json(
            directory / "result.json",
            {
                "schema": job["schema"],
                "id": job["id"],
                "files": files,
                "stats": stats,
                "seconds": time.monotonic() - started,
                "color": "scene-linear Rec.709",
                "alpha": "straight",
                "execution": execution,
            },
        )
        status(directory, "complete")
        return 0
    except BaseException as error:
        status(directory, "cancelled" if isinstance(error, InterruptedError) else "failed", str(error))
        return 1
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    parser.add_argument("--runtime", required=True)
    args = parser.parse_args()
    sys.exit(run(args.job, args.runtime))
