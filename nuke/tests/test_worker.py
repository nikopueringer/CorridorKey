"""Actual EXR/process tests using a synthetic CLI, not GPU quality claims."""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import OpenEXR
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ck2_nuke.client import run
from ck2_nuke.jobs import create_job, read_json, seal_job, verify_result
from ck2_nuke.worker import normalize


def make_job(tmp_path, first=1001, last=1003, **kwargs):
    directory = create_job(tmp_path, first, last, 8, 4, pixel_aspect=2.0, **kwargs)
    for i in range(last - first + 1):
        with OpenEXR.File({"pixelAspectRatio": 2.0}, {c: np.ones((4, 8), dtype=np.float32) for c in "RGB"}) as image:
            image.write(str(directory / ("source/%06d.exr" % i)))
    seal_job(directory)
    return directory


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / "runtime with 'quotes'"
    root.mkdir()
    (root / "infer.py").write_text("""
import argparse
from pathlib import Path
import numpy as np
import OpenEXR
p = argparse.ArgumentParser()
p.add_argument('--output_dir')
p.add_argument('--input')
args, _ = p.parse_known_args()
with OpenEXR.File(str(sorted(Path(args.input).glob('*.exr'))[0]), separate_channels=True) as image:
    shape = image.channels()['R'].pixels.shape
root = Path(args.output_dir) / 'source_cutout_exr'
root.mkdir(parents=True)
for i in range(3):
    channels = {c: np.full(shape, value, dtype=np.float32)
                for c,value in zip('RGBA', [4.5+i, -0.25, 0.75, 0.5])}
    with OpenEXR.File({}, channels) as image:
        image.write(str(root / ('cutout_%06d.exr' % i)))
""")
    return root


@pytest.mark.parametrize("model_scale", [1.0, 0.5])
def test_local_process_preserves_hdr_alpha_par_and_frame_range(tmp_path, runtime, model_scale):
    directory = make_job(tmp_path, model_scale=model_scale)
    assert run(directory, str(runtime), sys.executable) == 0
    result = verify_result(directory)
    assert len(result["files"]) == 3
    with OpenEXR.File(str(directory / "output/rgba.001003.exr"), separate_channels=True) as image:
        assert image.header()["pixelAspectRatio"] == 2.0
        assert np.allclose(image.channels()["R"].pixels, 6.5, atol=1e-6)
        assert np.allclose(image.channels()["G"].pixels, -0.25, atol=1e-6)
        assert np.allclose(image.channels()["A"].pixels, 0.5, atol=1e-6)
        assert "acesImageContainerFlag" not in image.header()
    # Re-running must not overwrite completed outputs.
    assert run(directory, str(runtime), sys.executable) == 1
    verify_result(directory)


def test_worker_rejects_missing_frame_and_nonfinite_pixels(tmp_path):
    directory = make_job(tmp_path, last=1001)
    raw = directory / "raw/source_cutout_exr"
    raw.mkdir(parents=True)
    with pytest.raises(ValueError, match="count"):
        normalize(directory, read_json(directory / "job.json"))
    with OpenEXR.File({}, {c: np.full((4, 8), np.nan, dtype=np.float32) for c in "RGBA"}) as image:
        image.write(str(raw / "cutout_000000.exr"))
    with pytest.raises(ValueError, match="Non-finite"):
        normalize(directory, read_json(directory / "job.json"))


def test_external_runtime_random_state_is_repeatable(tmp_path, runtime):
    script = runtime / "infer.py"
    script.write_text(script.read_text().replace("4.5+i", "4.5+i+np.random.random()+float(hash('ck2')%1000)/1000"))
    first, second = make_job(tmp_path), make_job(tmp_path)
    assert run(first, str(runtime), sys.executable) == 0
    assert run(second, str(runtime), sys.executable) == 0
    assert verify_result(first)["files"] == verify_result(second)["files"]


@pytest.mark.parametrize("cancel", [True, False])
def test_worker_stops_child_on_cancel_or_timeout(tmp_path, runtime, cancel):
    directory = make_job(tmp_path, timeout=1 if not cancel else 30)
    (runtime / "infer.py").write_text("import time\ntime.sleep(120)\n")
    worker = Path(__file__).resolve().parents[1] / "ck2_nuke/worker.py"
    process = subprocess.Popen([sys.executable, str(worker), str(directory), "--runtime", str(runtime)])
    try:
        deadline = time.monotonic() + 10
        while not (directory / "inference.log").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (directory / "inference.log").exists()
        if cancel:
            (directory / "CANCEL").touch()
        assert process.wait(timeout=10) == 1
        state = read_json(directory / "status.json")
        assert state["state"] == ("cancelled" if cancel else "failed")
        assert not (directory / "result.json").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
