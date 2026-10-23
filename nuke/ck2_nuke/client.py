"""Local/SSH job runner. Uses OpenSSH configuration and no listening server."""

import argparse
import re
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ck2_nuke.jobs import expected_inputs, extract_results, read_json, verify_inputs, verify_result, write_json

SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=3",
    "-o",
    "ControlMaster=no",
    "-o",
    "ControlPath=none",
]


def ssh_command(host, argv):
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", host):
        raise ValueError("Use an SSH hostname or configured alias, without options")
    return ["ssh"] + SSH_OPTIONS + [host, shlex.join([str(a) for a in argv])]


def set_status(directory, state, message=""):
    write_json(Path(directory) / "status.json", {"state": state, "message": message, "time": time.time()})


def wait_process(process, directory, timeout, cancel_remote=None):
    started = time.monotonic()
    try:
        while process.poll() is None:
            if (Path(directory) / "CANCEL").exists():
                if cancel_remote:
                    cancel_remote()
                # The worker owns its process group and shuts down inference first.
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.terminate()
                raise InterruptedError("Cancelled by user")
            if time.monotonic() - started > timeout:
                if cancel_remote:
                    cancel_remote()
                raise TimeoutError("Transport/worker timeout; see transport.log")
            time.sleep(0.25)
        return process.returncode
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def upload(directory, job, host, remote):
    # Only our generated relative paths go into this archive. No Nuke scripts,
    # SSH files, arbitrary directories or user Python callbacks are transferred.
    setup = "mkdir -p -- %s && mkdir -- %s && tar -xf - -C %s" % (
        shlex.quote(str(remote.parent)),
        shlex.quote(str(remote)),
        shlex.quote(str(remote)),
    )
    with (directory / "transport.log").open("ab") as log:
        process = subprocess.Popen(["ssh"] + SSH_OPTIONS + [host, setup], stdin=subprocess.PIPE, stdout=log, stderr=log)
        try:
            with tarfile.open(fileobj=process.stdin, mode="w|") as archive:
                for name in ["job.json"] + expected_inputs(job):
                    archive.add(str(directory / name), arcname=name, recursive=False)
                package = Path(__file__).parent
                for name in ("__init__.py", "jobs.py", "worker.py", "runtime_entry.py"):
                    archive.add(str(package / name), arcname="ck2_nuke/" + name, recursive=False)
            process.stdin.close()
            if wait_process(process, directory, job["timeout"]):
                raise RuntimeError("SSH upload failed; see transport.log")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()


def download(directory, host, remote):
    argv = ["tar", "-cf", "-", "-C", str(remote), "result.json", "inference.log", "status.json"]
    # Explicit files prevent directory entries/symlinks being accepted by our extractor.
    remote_job = read_json(directory / "job.json")
    argv += ["output/rgba.%06d.exr" % f for f in range(remote_job["first"], remote_job["last"] + 1)]
    with (directory / "transport.log").open("ab") as log:
        process = subprocess.Popen(ssh_command(host, argv), stdout=subprocess.PIPE, stderr=log)
        try:
            extract_results(process.stdout, directory)
            process.stdout.close()
            if process.wait(timeout=60):
                raise RuntimeError("SSH result download failed; see transport.log")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    verify_result(directory)


def run(directory, runtime, python, host=None, remote_root=None, retrieve=False):
    directory = Path(directory).expanduser().resolve()
    try:
        job = read_json(directory / "job.json")
        verify_inputs(directory, job)
        if not Path(runtime).is_absolute() or not Path(python).is_absolute():
            raise ValueError("Runtime and Python paths must be absolute on the execution machine")
        if (directory / "CANCEL").exists():
            raise InterruptedError("Cancelled by user")
        if host:
            ssh_command(host, ["true"])  # Validate before constructing the upload command.
            if not remote_root or not Path(remote_root).is_absolute():
                raise ValueError("Remote job root must be an absolute path")
            remote = Path(remote_root) / job["id"]
            write_json(directory / "remote.json", {"host": host, "directory": str(remote)})

            def cancel_remote():
                subprocess.run(ssh_command(host, ["touch", "--", str(remote / "CANCEL")]), timeout=25, check=True)

            if not retrieve:
                set_status(directory, "uploading")
                upload(directory, job, host, remote)
                set_status(directory, "running", "Remote CUDA inference; details in transport.log")
                with (directory / "transport.log").open("ab") as log:
                    process = subprocess.Popen(
                        ssh_command(
                            host, [python, "-u", str(remote / "ck2_nuke/worker.py"), str(remote), "--runtime", runtime]
                        ),
                        stdout=log,
                        stderr=log,
                    )
                    if wait_process(process, directory, job["timeout"] + 60, cancel_remote):
                        raise RuntimeError("Remote worker failed; see transport.log and remote inference.log")
            set_status(directory, "downloading")
            download(directory, host, remote)
        else:
            if retrieve:
                raise ValueError("--retrieve requires SSH mode")
            worker = Path(__file__).with_name("worker.py")
            set_status(directory, "starting")
            with (directory / "transport.log").open("ab") as log:
                process = subprocess.Popen(
                    [python, "-u", str(worker), str(directory), "--runtime", runtime], stdout=log, stderr=log
                )
                if wait_process(process, directory, job["timeout"] + 60):
                    raise RuntimeError("Local worker failed; see status.json and inference.log")
            verify_result(directory)
        set_status(directory, "complete", "Validated RGBA EXRs are ready")
        return 0
    except Exception as error:
        set_status(directory, "cancelled" if isinstance(error, InterruptedError) else "failed", str(error))
        print(str(error), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--host")
    parser.add_argument("--remote-root")
    parser.add_argument("--retrieve", action="store_true", help="Recover a completed SSH job without re-running it")
    args = parser.parse_args()
    sys.exit(run(args.job, args.runtime, args.python, args.host, args.remote_root, args.retrieve))
