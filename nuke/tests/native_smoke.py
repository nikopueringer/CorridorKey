"""Run in Nuke -t with CK2_TEST_ROOT and optionally CK2_TEST_RUNTIME/PYTHON.

Creates a separate synthetic graph. No user script or startup file is edited.
With a runtime configured, performs a bounded three-frame GPU bake.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import nuke

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ck2_nuke import ui
from ck2_nuke.jobs import create_job, read_json, seal_job

root = Path(os.environ["CK2_TEST_ROOT"]).resolve()
root.mkdir(parents=True, exist_ok=True)
nuke.root()["first_frame"].setValue(1001)
nuke.root()["last_frame"].setValue(1003)
nuke.root()["format"].setValue(nuke.addFormat("640 360 2 ck2_smoke"))
plate = nuke.nodes.Constant()
plate["color"].setValue([2.0, -0.25, 0.5, 0.5])
group = ui.create_node()
group.setInput(0, plate)
group["ck_jobs"].setValue(str(root / "jobs"))
job_dir = create_job(root / "export", 1001, 1003, 640, 360, pixel_aspect=2.0)
ui._export(plate, job_dir / "source", 1001, 1003, "linear")
seal_job(job_dir)
read = nuke.nodes.Read(file=str(job_dir / "source/%06d.exr"), first=0, last=2, colorspace="linear")
assert (read.width(), read.height(), read.format().pixelAspect()) == (640, 360, 2.0)
assert abs(read.sample("red", 100, 100) - 2.0) < 1e-6
assert abs(read.sample("green", 100, 100) + 0.25) < 1e-6
assert not group["ck_use_cache"].value()
assert not group["ck_first"].enabled()
group["ck_range"].setValue("Custom")
ui.knob_changed(group, group["ck_range"])
assert group["ck_first"].enabled()
group["ck_first"].setValue(1001)
group["ck_last"].setValue(1003)
group["ck_scale"].setValue("Quarter")

# Verify hint extraction from alpha, rather than reading the RGB channels.
hint_job = create_job(root / "export", 1001, 1001, 640, 360, pixel_aspect=2.0, hint="first")
ui._export(plate, hint_job / "hint", 1001, 1001, "linear", True)
hint = nuke.nodes.Read(file=str(hint_job / "hint/000000.exr"), colorspace="linear")
assert abs(hint.sample("red", 100, 100) - 0.5) < 1e-6

# Verify exports from inside a nested Group without cross-context connections.
nested = nuke.nodes.Group(name="NestedSmoke")
with nested:
    nested_plate = nuke.nodes.Constant()
    nested_plate["color"].setValue([3.0, 0.0, 0.0, 1.0])
nested_dir = create_job(root / "export", 1001, 1001, 640, 360, pixel_aspect=2.0)
ui._export(nested_plate, nested_dir / "source", 1001, 1001, "linear")
nested_read = nuke.nodes.Read(file=str(nested_dir / "source/000000.exr"), colorspace="linear")
assert abs(nested_read.sample("red", 100, 100) - 3.0) < 1e-6

report = {
    "nuke": nuke.NUKE_VERSION_STRING,
    "export_job": str(job_dir),
    "hdr_negative_preserved": True,
    "hint_from_alpha": True,
    "pixel_aspect": 2.0,
}
runtime = os.environ.get("CK2_TEST_RUNTIME")
if runtime:
    source_path = os.environ.get("CK2_TEST_SOURCE")
    if source_path:
        source = nuke.nodes.Read(file=source_path, first=1001, last=1003, colorspace="linear")
        resize = nuke.nodes.Reformat(inputs=[source])
        resize["format"].setValue(nuke.addFormat("640 440 2 ck2_smoke_plate"))
        group.setInput(0, resize)
    group["ck_python"].setValue(os.environ["CK2_TEST_PYTHON"])
    group["ck_runtime"].setValue(runtime)
    group["ck_mode"].setValue("Local")
    group["ck_client_python"].setValue(os.environ["CK2_TEST_PYTHON"])
    probe = (
        "import os,json,torch,cv2,ctypes; "
        "print(json.dumps({'rounding':ctypes.CDLL(None).fegetround(),"
        "'tf32':torch.backends.cuda.matmul.allow_tf32,"
        "'cudnn_tf32':torch.backends.cudnn.allow_tf32,'threads':torch.get_num_threads(),"
        "'precision':torch.get_float32_matmul_precision(),'cv2':cv2.__version__,"
        "'env':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS',"
        "'OPENBLAS_NUM_THREADS','PYTHONHASHSEED','NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',"
        "'LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME']}}))"
    )
    report["numeric_environment"] = json.loads(
        subprocess.check_output([os.environ["CK2_TEST_PYTHON"], "-c", probe], text=True)
    )
    print("NUMERIC_ENVIRONMENT", json.dumps(report["numeric_environment"]), flush=True)
    directory = os.environ.get("CK2_TEST_LOAD_JOB")
    if directory:
        group["ck_job"].setValue(directory)
        ui.load_result(group)
    else:
        directory = ui.submit(group, blocking=True)
    report["inference_job"] = directory
    print(
        "NATIVE_GRAPH_FORMATS",
        json.dumps(
            {
                "group": [group.width(), group.height(), group.format().pixelAspect()],
                "read": [
                    group.node("CK2_Result").width(),
                    group.node("CK2_Result").height(),
                    group.node("CK2_Result").format().pixelAspect(),
                ],
                "which": group.node("CK2_Cache")["which"].value(),
                "cache": group["ck_use_cache"].value(),
            }
        ),
        flush=True,
    )
    nuke.scriptSaveAs(str(root / "CK2_smoke.nk"), overwrite=1)
    assert group["ck_use_cache"].value()
    assert group.width() == 640
    assert group.format().pixelAspect() == read_json(Path(directory) / "job.json")["pixel_aspect"]
    nuke.frame(1002)
    raw = group.node("CK2_Result")
    for channel in ("red", "green", "blue"):
        expected = raw.sample(channel, 320, 220) * raw.sample("alpha", 320, 220)
        assert abs(group.sample(channel, 320, 220) - expected) < 1e-5

    # Exercise the actual Qt timer/main-thread callback without another GPU run.
    from PySide2 import QtCore

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    for changed in (False, True):
        group["ck_use_cache"].setValue(False)
        process = subprocess.Popen(["/usr/bin/python3", "-c", "pass"])
        ui.monitor(group, process, Path(directory))
        if changed:
            group["ck_hint_quality"].setValue(0.5)
        deadline = time.monotonic() + 5
        while group.fullName() in ui._RUNNING and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.02)
        assert group.fullName() not in ui._RUNNING
        assert group["ck_use_cache"].value() == (not changed)
    ui.load_result(group)
    report["qt_completion_and_changed_settings_guard"] = True
    report["native_premult_check"] = True
    report["result"] = read_json(Path(directory) / "result.json")
    # Native downstream render proves that the loaded result is usable in a graph.
    write = nuke.nodes.Write(inputs=[group], file=str(root / "nuke_result.%04d.exr"), file_type="exr")
    write["channels"].setValue("rgba")
    write["datatype"].setValue("32 bit float")
    nuke.execute(write, 1001, 1003)
    nuke.scriptSaveAs(str(root / "CK2_smoke.nk"), overwrite=1)

(root / "native_report.json").write_text(json.dumps(report, indent=2) + "\n")
print("CK2_NUKE_NATIVE_SMOKE_PASSED", json.dumps(report), flush=True)
