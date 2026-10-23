"""Portable, dependency-free job and transport contracts."""

import hashlib
import json
import math
import os
import re
import tarfile
import uuid
from pathlib import Path

SCHEMA = 1


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def create_job(
    root,
    first,
    last,
    width,
    height,
    pixel_aspect=1.0,
    fps=24.0,
    colorspace="linear",
    hint="none",
    hint_quality=0.75,
    model_scale=1.0,
    low_vram=False,
    despill=0.0,
    timeout=7200,
):
    job = {
        "schema": SCHEMA,
        "id": uuid.uuid4().hex,
        "first": first,
        "last": last,
        "width": width,
        "height": height,
        "pixel_aspect": pixel_aspect,
        "fps": fps,
        "colorspace": colorspace,
        "hint": hint,
        "hint_quality": hint_quality,
        "model_scale": model_scale,
        "low_vram": low_vram,
        "despill": despill,
        "timeout": timeout,
    }
    validate_job(job)
    directory = Path(root).expanduser().resolve() / job["id"]
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "source").mkdir()
    (directory / "hint").mkdir()
    write_json(directory / "job.json", job)
    return directory


def validate_job(job):
    if job.get("schema") != SCHEMA or not re.fullmatch(r"[0-9a-f]{32}", job.get("id", "")):
        raise ValueError("Unsupported job schema or invalid job ID")
    for field in ("first", "last", "width", "height"):
        if type(job.get(field)) is not int:
            raise ValueError("Job %s must be an integer" % field)
    if job["last"] < job["first"] or min(job["width"], job["height"]) < 1:
        raise ValueError("Invalid frame range or image size")
    for name in ("pixel_aspect", "fps", "hint_quality", "model_scale", "despill", "timeout"):
        value = job.get(name)
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError("Invalid %s" % name)
    if min(job["pixel_aspect"], job["fps"], job["timeout"]) <= 0:
        raise ValueError("Pixel aspect, FPS and timeout must be positive")
    if not 0 < job["model_scale"] <= 1:
        raise ValueError("Model scale must be in (0, 1]")
    if not 0 <= job["hint_quality"] <= 1 or not 0 <= job["despill"] <= 1:
        raise ValueError("Hint quality and despill must be in [0, 1]")
    if job.get("hint") not in ("none", "first", "all"):
        raise ValueError("Invalid hint mode")
    if type(job.get("low_vram")) is not bool:
        raise ValueError("Invalid low_vram flag")
    if not isinstance(job.get("colorspace"), str) or not job["colorspace"]:
        raise ValueError("An explicit linear Rec.709 colour space is required")


def expected_inputs(job):
    count = job["last"] - job["first"] + 1
    source = ["source/%06d.exr" % i for i in range(count)]
    hint_count = {"none": 0, "first": 1, "all": count}[job["hint"]]
    return source + ["hint/%06d.exr" % i for i in range(hint_count)]


def seal_job(directory):
    directory = Path(directory)
    job = read_json(directory / "job.json")
    validate_job(job)
    job["inputs"] = {name: digest(directory / name) for name in expected_inputs(job)}
    write_json(directory / "job.json", job)
    return job


def verify_inputs(directory, job):
    validate_job(job)
    if set(job.get("inputs", {})) != set(expected_inputs(job)):
        raise ValueError("Input manifest does not match the requested range/hints")
    actual = set()
    for folder in ("source", "hint"):
        input_directory = Path(directory) / folder
        if not input_directory.exists():
            continue  # Empty optional hint directories need not be transported.
        for path in input_directory.iterdir():
            if not path.is_file() or path.is_symlink():
                raise ValueError("Inputs must be regular files")
            actual.add(folder + "/" + path.name)
    if actual != set(expected_inputs(job)):
        raise ValueError("Input directory has unexpected or missing files")
    for name, checksum in job["inputs"].items():
        if digest(Path(directory) / name) != checksum:
            raise ValueError("Input checksum mismatch: " + name)


def verify_result(directory):
    directory = Path(directory)
    job = read_json(directory / "job.json")
    validate_job(job)
    result = read_json(directory / "result.json")
    names = ["output/rgba.%06d.exr" % f for f in range(job["first"], job["last"] + 1)]
    if result.get("id") != job["id"] or result.get("schema") != SCHEMA or set(result.get("files", {})) != set(names):
        raise ValueError("Result does not match the submitted job")
    for name in names:
        if digest(directory / name) != result["files"][name]:
            raise ValueError("Output checksum mismatch: " + name)
    return result


def extract_results(stream, directory):
    """Accept only regular result files. Never unpack paths or links from a peer."""
    directory = Path(directory)
    seen = set()
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            allowed = name in ("result.json", "inference.log", "status.json") or re.fullmatch(
                r"output/rgba\.-?\d{5,}\.exr", name
            )
            if not member.isfile() or not allowed or name in seen:
                raise ValueError("Unexpected result archive member: " + name)
            seen.add(name)
            destination = directory / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".incoming")
            with archive.extractfile(member) as source, temporary.open("wb") as target:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    target.write(block)
            os.replace(str(temporary), str(destination))
