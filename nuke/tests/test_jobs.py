"""Contract tests run without Nuke, CUDA or the CK2 model."""

import io
import shlex
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ck2_nuke.client import run, ssh_command
from ck2_nuke.jobs import (
    create_job,
    digest,
    extract_results,
    read_json,
    seal_job,
    verify_inputs,
    verify_result,
    write_json,
)
from ck2_nuke.worker import command


def job_fixture(tmp_path, **kwargs):
    directory = create_job(tmp_path, 1001, 1003, 640, 360, pixel_aspect=2.0, **kwargs)
    for i in range(3):
        (directory / ("source/%06d.exr" % i)).write_bytes(b"synthetic contract fixture")
    return directory


def test_seal_detects_changed_input_before_start(tmp_path):
    directory = job_fixture(tmp_path)
    job = seal_job(directory)
    verify_inputs(directory, job)
    (directory / "source/000001.exr").write_bytes(b"changed")
    assert run(directory, "/no/runtime", sys.executable) == 1
    assert "checksum mismatch" in read_json(directory / "status.json")["message"]
    assert not (directory / "raw").exists()


def test_missing_source_frame_is_not_silently_skipped(tmp_path):
    directory = job_fixture(tmp_path)
    (directory / "source/000001.exr").unlink()
    with pytest.raises(FileNotFoundError):
        seal_job(directory)


def test_unlisted_input_cannot_be_processed(tmp_path):
    directory = job_fixture(tmp_path)
    job = seal_job(directory)
    (directory / "source/unlisted.exr").write_bytes(b"extra frame")
    with pytest.raises(ValueError, match="unexpected"):
        verify_inputs(directory, job)


def test_transport_may_omit_empty_optional_hint_directory(tmp_path):
    directory = job_fixture(tmp_path)
    job = seal_job(directory)
    (directory / "hint").rmdir()
    verify_inputs(directory, job)


@pytest.mark.parametrize(
    "hint,flag,count", [("none", None, 0), ("first", "--hint_first_frame", 1), ("all", "--hint_video", 3)]
)
def test_hint_alignment_and_linear_hdr_command(tmp_path, hint, flag, count):
    directory = job_fixture(tmp_path, hint=hint)
    for i in range(count):
        (directory / ("hint/%06d.exr" % i)).write_bytes(b"matte")
    job = seal_job(directory)
    argv = command(directory, "/runtime with spaces", job)
    assert "--frame_dir_linear" in argv and "--cutout_linear" in argv
    assert argv[argv.index("--exr_dtype") + 1] == "float"
    assert ("--hint_video" in argv) == (hint == "all")
    assert ("--hint_first_frame" in argv) == (hint == "first")
    if flag:
        assert str(directory / "hint") in argv[argv.index(flag) + 1]
    assert "--hann_chunk" not in argv  # Let CK2 measure free VRAM.


@pytest.mark.parametrize(
    "options", [{"model_scale": 0}, {"model_scale": float("nan")}, {"hint": "other"}, {"fps": -1}, {"hint_quality": 2}]
)
def test_reject_invalid_jobs_before_creating_directory(tmp_path, options):
    with pytest.raises(ValueError):
        job_fixture(tmp_path, **options)
    assert not list(tmp_path.iterdir())


def test_result_frame_numbers_and_checksums(tmp_path):
    directory = job_fixture(tmp_path)
    job = seal_job(directory)
    (directory / "output").mkdir()
    files = {}
    for frame in range(1001, 1004):
        name = "output/rgba.%06d.exr" % frame
        (directory / name).write_bytes(str(frame).encode())
        files[name] = digest(directory / name)
    write_json(directory / "result.json", {"schema": 1, "id": job["id"], "files": files})
    verify_result(directory)
    (directory / "output/rgba.001002.exr").write_bytes(b"corrupted in transit")
    with pytest.raises(ValueError, match="checksum"):
        verify_result(directory)


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escape", tarfile.REGTYPE),
        ("/absolute", tarfile.REGTYPE),
        ("output/rgba.001001.exr", tarfile.SYMTYPE),
        ("ck2_nuke/worker.py", tarfile.REGTYPE),
    ],
)
def test_result_archive_cannot_write_code_or_escape(tmp_path, name, kind):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.size = 1 if kind == tarfile.REGTYPE else 0
        member.linkname = "/tmp/escape" if kind == tarfile.SYMTYPE else ""
        archive.addfile(member, io.BytesIO(b"x") if member.size else None)
    data.seek(0)
    with pytest.raises(ValueError, match="Unexpected"):
        extract_results(data, tmp_path)
    assert not list(tmp_path.iterdir())


def test_ssh_quotes_remote_arguments_and_rejects_option_injection():
    argv = ["/python with spaces", "--runtime", "/path/with 'quotes' and $(touch bad)"]
    encoded = ssh_command("user@gpu-alias", argv)
    assert shlex.split(encoded[-1]) == argv
    for bad in ("-oProxyCommand=bad", "host;touch bad", "host\nother"):
        with pytest.raises(ValueError):
            ssh_command(bad, argv)


def test_cancelled_submission_does_not_launch(tmp_path):
    directory = job_fixture(tmp_path)
    seal_job(directory)
    (directory / "CANCEL").touch()
    assert run(directory, "/runtime", sys.executable) == 1
    assert read_json(directory / "status.json")["state"] == "cancelled"
