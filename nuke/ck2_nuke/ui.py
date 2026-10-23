"""Nuke 15+ Group front end. GPU inference is a deliberate disk bake."""

import os
import subprocess
from pathlib import Path

import nuke
from ck2_nuke.jobs import create_job, read_json, seal_job, verify_result

_RUNNING = {}


def _knob(node, knob, default=None):
    return node[knob].value() if knob in node.knobs() else default


def _settings(group):
    names = (
        "ck_range",
        "ck_first",
        "ck_last",
        "ck_hint",
        "ck_hint_quality",
        "ck_scale",
        "ck_low_vram",
        "ck_despill",
        "ck_colorspace",
    )
    return tuple(group[name].value() for name in names) + tuple(
        group.input(i).fullName() if group.input(i) else None for i in range(2)
    )


def create_node():
    selected = nuke.selectedNodes()
    source = selected[-1] if selected else None
    group = nuke.nodes.Group(name="CorridorKey2", label="Baked key | Local / SSH")
    group.begin()
    try:
        plate = nuke.nodes.Input(name="Source", number=0)
        nuke.nodes.Input(name="AlphaHint", number=1)
        result = nuke.nodes.Read(name="CK2_Result")
        premult = nuke.nodes.Premult(name="CK2_Premult", inputs=[result])
        switch = nuke.nodes.Switch(name="CK2_Cache", inputs=[plate, premult])
        nuke.nodes.Output(inputs=[switch])
    finally:
        group.end()
    if source:
        group.setInput(0, source)
    group.addKnob(nuke.Tab_Knob("ck2", "CorridorKey 2"))
    mode = nuke.Enumeration_Knob("ck_mode", "Compute", ["Local", "SSH"])
    mode.setValue("SSH" if os.environ.get("CK2_REMOTE_HOST") else "Local")
    group.addKnob(mode)
    group.addKnob(nuke.Enumeration_Knob("ck_range", "Range", ["Source", "Custom"]))
    for name, label, value in (
        ("ck_first", "First", int(nuke.root().firstFrame())),
        ("ck_last", "Last", int(nuke.root().lastFrame())),
    ):
        knob = nuke.Int_Knob(name, label)
        knob.setValue(value)
        group.addKnob(knob)
    group.addKnob(nuke.Enumeration_Knob("ck_hint", "Alpha hint", ["None", "First frame", "All frames"]))
    for name, label, value in (("ck_hint_quality", "Hint strength", 0.75), ("ck_despill", "Despill", 0.0)):
        knob = nuke.Double_Knob(name, label)
        knob.setRange(0, 1)
        knob.setValue(value)
        group.addKnob(knob)
    scale = nuke.Enumeration_Knob("ck_scale", "Internal resolution", ["Full", "Half", "Quarter"])
    scale.setValue("Half")
    group.addKnob(scale)
    group.addKnob(nuke.Boolean_Knob("ck_low_vram", "Low GPU memory"))
    cache = nuke.Boolean_Knob("ck_use_cache", "Use baked result")
    cache.setTooltip("A disk bake. Render again after changing upstream nodes or settings.")
    group.addKnob(cache)
    group.node("CK2_Cache")["which"].setExpression("parent.ck_use_cache")
    for name, label, action in (
        ("ck_render", "Render range", "submit(nuke.thisNode())"),
        ("ck_cancel", "Cancel", "cancel(nuke.thisNode())"),
        ("ck_load", "Load completed result", "load_result(nuke.thisNode())"),
    ):
        group.addKnob(nuke.PyScript_Knob(name, label, "import ck2_nuke.ui; ck2_nuke.ui." + action))
    group.addKnob(nuke.String_Knob("ck_status", "Status"))
    group["ck_status"].setValue("Ready. Source range; optional hint on input 1.")
    group["ck_status"].setEnabled(False)
    group.addKnob(nuke.Tab_Knob("ck_settings", "Setup"))
    settings = (
        ("ck_jobs", "Local job folder", os.environ.get("CK2_JOBS", str(Path.home() / "ck2_jobs"))),
        ("ck_colorspace", "Linear Rec.709 colour space", os.environ.get("CK2_COLORSPACE", "linear")),
        ("ck_client_python", "Client Python (3.9+)", os.environ.get("CK2_CLIENT_PYTHON", "")),
        ("ck_python", "Local CUDA Python", os.environ.get("CK2_PYTHON", "")),
        ("ck_runtime", "Local CK2 runtime", os.environ.get("CK2_RUNTIME", "")),
        ("ck_host", "SSH alias", os.environ.get("CK2_REMOTE_HOST", "")),
        ("ck_remote_python", "Remote CUDA Python", os.environ.get("CK2_REMOTE_PYTHON", "")),
        ("ck_remote_runtime", "Remote CK2 runtime", os.environ.get("CK2_REMOTE_RUNTIME", "")),
        ("ck_remote_root", "Remote job folder", os.environ.get("CK2_REMOTE_JOBS", "")),
    )
    for name, label, value in settings:
        knob = nuke.String_Knob(name, label)
        knob.setValue(value)
        group.addKnob(knob)
    group["ck_colorspace"].setTooltip(
        "Must encode scene-linear Rec.709/sRGB primaries. Default Nuke: linear. "
        "ACES 1.x commonly: Utility - Linear - sRGB. Choose the matching space in your OCIO config."
    )
    group.addKnob(
        nuke.Text_Knob(
            "ck_note",
            "",
            "CK2 runtime is installed separately. Output: premultiplied RGBA.\n"
            "Render again after upstream edits. Export uses the full format rectangle.",
        )
    )
    path = nuke.String_Knob("ck_job", "Job")
    path.setVisible(False)
    group.addKnob(path)
    group["knobChanged"].setValue("import ck2_nuke.ui; ck2_nuke.ui.knob_changed(nuke.thisNode(), nuke.thisKnob())")
    knob_changed(group, group["ck_range"])
    return group


def knob_changed(group, knob):
    if knob is None or "ck_range" not in group.knobs():
        return
    if knob.name() == "ck_range":
        for name in ("ck_first", "ck_last"):
            group[name].setEnabled(group["ck_range"].value() == "Custom")
    if (
        knob.name()
        in (
            "ck_range",
            "ck_first",
            "ck_last",
            "ck_hint",
            "ck_hint_quality",
            "ck_scale",
            "ck_low_vram",
            "ck_despill",
            "ck_colorspace",
        )
        and "ck_use_cache" in group.knobs()
    ):
        group["ck_use_cache"].setValue(False)


def _export(source, directory, first, last, colorspace, hint=False):
    parent_name = source.fullName().rpartition(".")[0]
    parent = nuke.toNode(parent_name) if parent_name else nuke.root()
    with parent:
        _export_nodes(source, directory, first, last, colorspace, hint)


def _export_nodes(source, directory, first, last, colorspace, hint):
    temporary = []
    try:
        upstream = source
        if hint:
            shuffle = nuke.nodes.Shuffle(inputs=[source])
            temporary.append(shuffle)
            for channel in ("red", "green", "blue", "alpha"):
                shuffle[channel].setValue("alpha")
            upstream = shuffle
        crop = nuke.nodes.Crop(inputs=[upstream])
        temporary.append(crop)
        fmt = source.format()
        crop["box"].setValue([0, 0, fmt.width(), fmt.height()])
        crop["reformat"].setValue(True)
        write = nuke.nodes.Write(inputs=[crop])
        temporary.append(write)
        write["file"].setValue(str(Path(directory) / "export.%06d.exr"))
        write["file_type"].setValue("exr")
        write["channels"].setValue("rgb")
        write["datatype"].setValue("32 bit float")
        if hint:
            write["raw"].setValue(True)
        else:
            write["colorspace"].setValue(colorspace)
        nuke.execute(write, first, last)
        for index, frame in enumerate(range(first, last + 1)):
            (Path(directory) / ("export.%06d.exr" % frame)).rename(Path(directory) / ("%06d.exr" % index))
    finally:
        for node in reversed(temporary):
            nuke.delete(node)


def submit(group, blocking=False):
    if group.fullName() in _RUNNING:
        raise RuntimeError("This node already has a running job")
    source = group.input(0)
    if source is None:
        raise ValueError("Connect the source plate to input 0")
    hint = {"None": "none", "First frame": "first", "All frames": "all"}[group["ck_hint"].value()]
    if hint != "none" and group.input(1) is None:
        raise ValueError("Connect an alpha hint to input 1")
    if group["ck_range"].value() == "Source":
        first, last = int(source.firstFrame()), int(source.lastFrame())
    else:
        first, last = int(group["ck_first"].value()), int(group["ck_last"].value())
    remote = group["ck_mode"].value() == "SSH"
    python = group["ck_remote_python" if remote else "ck_python"].value()
    runtime = group["ck_remote_runtime" if remote else "ck_runtime"].value()
    client_python = group["ck_client_python"].value() or ("/usr/bin/python3" if remote else python)
    if not Path(python).is_absolute() or not Path(runtime).is_absolute() or not Path(client_python).is_file():
        raise ValueError("Set absolute runtime/Python paths on the Setup tab")
    if remote and (not group["ck_host"].value() or not Path(group["ck_remote_root"].value()).is_absolute()):
        raise ValueError("Set the SSH alias and absolute remote job folder on Setup")
    fmt = source.format()
    if hint != "none":
        other = group.input(1).format()
        if (other.width(), other.height(), other.pixelAspect()) != (fmt.width(), fmt.height(), fmt.pixelAspect()):
            raise ValueError("Hint format and pixel aspect must match the source")
    colorspace = group["ck_colorspace"].value()
    directory = create_job(
        group["ck_jobs"].value(),
        first,
        last,
        fmt.width(),
        fmt.height(),
        fmt.pixelAspect(),
        nuke.root()["fps"].value(),
        colorspace,
        hint,
        group["ck_hint_quality"].value(),
        {"Full": 1.0, "Half": 0.5, "Quarter": 0.25}[group["ck_scale"].value()],
        group["ck_low_vram"].value(),
        group["ck_despill"].value(),
    )
    group["ck_job"].setValue(str(directory))
    group["ck_use_cache"].setValue(False)
    group["ck_status"].setValue("Exporting frames")
    try:
        _export(source, directory / "source", first, last, colorspace)
        if hint != "none":
            _export(group.input(1), directory / "hint", first, first if hint == "first" else last, colorspace, True)
        seal_job(directory)
    except BaseException:
        group["ck_status"].setValue("Export failed/cancelled; see job folder")
        raise
    argv = [
        client_python,
        "-u",
        str(Path(__file__).with_name("client.py")),
        str(directory),
        "--runtime",
        runtime,
        "--python",
        python,
    ]
    if remote:
        argv += ["--host", group["ck_host"].value(), "--remote-root", group["ck_remote_root"].value()]
    group["ck_status"].setValue("Starting inference")
    if blocking or not nuke.GUI:
        with (directory / "client.log").open("wb") as log:
            result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            group["ck_status"].setValue(read_json(directory / "status.json")["message"])
            raise RuntimeError("CK2 failed; see " + str(directory))
        load_result(group)
    else:
        with (directory / "client.log").open("wb") as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        monitor(group, process, directory)
    return str(directory)


def monitor(group, process, directory):
    try:
        from PySide2 import QtCore
    except ImportError:
        from PySide6 import QtCore

    name, submitted = group.fullName(), _settings(group)
    timer = QtCore.QTimer()

    def poll():
        node = nuke.toNode(name)
        if node is not None and _knob(node, "ck_job") == str(directory):
            if (directory / "status.json").exists():
                progress = read_json(directory / "status.json")
                node["ck_status"].setValue(progress["state"] + ": " + progress["message"])
        if process.poll() is not None:
            timer.stop()
            _RUNNING.pop(name, None)
            if process.returncode == 0 and node is not None and _knob(node, "ck_job") == str(directory):
                if _settings(node) == submitted:
                    try:
                        load_result(node)
                    except Exception as error:
                        node["ck_status"].setValue("Result load failed: " + str(error))
                else:
                    node["ck_status"].setValue("Result ready; settings changed. Use Load completed result.")

    timer.timeout.connect(poll)
    _RUNNING[name] = (process, timer, directory)
    timer.start(1000)


def cancel(group):
    directory = group["ck_job"].value()
    if directory and Path(directory).is_dir():
        (Path(directory) / "CANCEL").touch()
        group["ck_status"].setValue("Cancelling")


def load_result(group):
    directory = Path(group["ck_job"].value())
    verify_result(directory)
    job = read_json(directory / "job.json")
    read = group.node("CK2_Result")
    read["file"].setValue(str(directory / "output/rgba.%06d.exr"))
    for knob, value in (
        ("first", job["first"]),
        ("last", job["last"]),
        ("origfirst", job["first"]),
        ("origlast", job["last"]),
    ):
        read[knob].setValue(value)
    read["premultiplied"].setValue(False)
    read["colorspace"].setValue(job["colorspace"])
    read["reload"].execute()
    fmt = read.format()
    if (fmt.width(), fmt.height(), fmt.pixelAspect()) != (job["width"], job["height"], job["pixel_aspect"]):
        raise ValueError("Nuke readback format/PAR does not match the submitted job")
    group["ck_use_cache"].setValue(True)
    group["ck_status"].setValue("Baked result loaded. Render again after upstream edits.")
