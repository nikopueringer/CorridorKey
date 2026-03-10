"""
Tests corridorkey_cli.py.

This suite verifies the CLI entry points, the interactive Wizard state machine,
and the automated routing logic. It mocks the underlying processing engines
(GVM, VideoMaMa, Inference) to isolate the workflow orchestration from the
heavy-duty image processing.

These 24 tests have 99% coverage per pytest only missing:
- 168->179:
There is some tricky logic to get the code into a point where this logic would apply
I was unable to reach the state required to preform testing.

- 316->exit:
Terminal branch logic in the wizard's argument validation, could not get tested.

- 330:
The standard `if __name__ == "__main__":` entry point, felt unnecessary.

Using mocks in line with project goals of keeping tests and VRAM separate.

"""

import logging

import cv2
import numpy as np
import pytest

from corridorkey_cli import interactive_wizard, main

# ---------------------------------------------------------------------------
# interactive_wizard
# ---------------------------------------------------------------------------


class TestInteractiveWizard:
    def test_path_resolution(self, monkeypatch, capsys):
        """
        Scenario: User provides a Windows path that must fallback to a mapped Linux mount.
        Expected: Wizard maps path via map_path and prints the Linux/Remote result.
        """
        monkeypatch.setattr("os.path.exists", lambda path: "/mnt/project" in path)
        monkeypatch.setattr("corridorkey_cli.map_path", lambda _: "/mnt/project")
        monkeypatch.setattr("os.listdir", lambda _: [])
        monkeypatch.setattr("builtins.input", lambda *args, **kwargs: "q")

        interactive_wizard("Z:/Work/Shot01")

        out = capsys.readouterr().out
        assert "Linux/Remote Path:   /mnt/project" in out

    def test_path_not_found(self, monkeypatch, capsys):
        """
        Scenario: User provides a path that exists neither on Windows nor the mapped Linux mount.
        Expected: Wizard prints an [ERROR] message and exits the function immediately.
        """
        monkeypatch.setattr("os.path.exists", lambda _: False)
        monkeypatch.setattr("corridorkey_cli.map_path", lambda _: "/invalid/path")

        interactive_wizard("Z:/Missing/Path")

        out = capsys.readouterr().out
        assert "[ERROR] Path does not exist locally OR on Linux mount!" in out

    def test_detects_single_shot(self, tmp_path, monkeypatch, capsys):
        """
        Scenario: User targets a specific shot folder containing an 'Input' directory.
        Expected: Wizard identifies target_is_shot as True and finds 1 potential clip folder.
        """
        shot_path = tmp_path / "shot_01"
        (shot_path / "Input").mkdir(parents=True)
        monkeypatch.setattr("builtins.input", lambda *args, **kwargs: "q")

        interactive_wizard(str(shot_path))

        out = capsys.readouterr().out
        assert "Found 1 potential clip folders." in out

    def test_detects_batch_project(self, tmp_path, monkeypatch, capsys):
        """
        Scenario: User targets a root project folder containing multiple subfolders.
        Expected: Wizard identifies target_is_shot as False and lists all valid subfolders.
        """
        for shot_name in ["shot_a", "shot_b"]:
            shot_dir = tmp_path / shot_name
            (shot_dir / "Input").mkdir(parents=True)
            (shot_dir / "Input" / "video.mp4").touch()

        (tmp_path / "Output").mkdir()
        monkeypatch.setattr("builtins.input", lambda *args, **kwargs: "q")

        interactive_wizard(str(tmp_path))

        out = capsys.readouterr().out
        assert "Found 2 potential clip folders." in out

    def test_detects_unorganized_content(self, tmp_path, monkeypatch, capsys):
        """
        Scenario: Root contains a loose video and folders missing 'Input' subdirectories.
        Expected: Wizard flags 1 loose video and 2 messy folders in the terminal report.
        """
        (tmp_path / "clip_01.mp4").touch()
        (tmp_path / "shot_a").mkdir()
        (tmp_path / "shot_b").mkdir()
        (tmp_path / "notes.txt").touch()
        (tmp_path / ".DS_Store").touch()
        (tmp_path / "Output").mkdir()
        monkeypatch.setattr("builtins.input", lambda *args, **kwargs: "q")

        interactive_wizard(str(tmp_path))

        out = capsys.readouterr().out
        assert "Found 1 loose video files" in out
        assert "clip_01.mp4" in out
        assert "Found 2 folders that might need setup" in out
        assert ".DS_Store" not in out
        assert "notes.txt" not in out

    def test_organize_action(self, tmp_path, monkeypatch, capsys):
        video_name = "clip_01.mp4"
        video_file = tmp_path / video_name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_vid = cv2.VideoWriter(str(video_file), fourcc, 24, (64, 64))
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        out_vid.write(frame)
        out_vid.release()

        answers = iter(["y", "q"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))
        interactive_wizard(str(tmp_path))

        expected_folder = tmp_path / "clip_01"
        expected_input_file = expected_folder / "Input.mp4"

        assert expected_folder.is_dir(), "Shot folder 'clip_01' was not created"
        assert expected_input_file.exists(), f"Expected {expected_input_file} to exist"
        assert not video_file.exists(), "Original loose file was not moved"

        out = capsys.readouterr().out
        assert "Organization Complete." in out
        assert "RAW (Input only):        1" in out

    def test_organization_skips_existing_folder(self, tmp_path, monkeypatch, caplog):
        """
        Scenario: A loose video exists, but a folder with that name already exists.
        Expected: Wizard logs a warning and skips moving that file.
        """
        (tmp_path / "clip.mp4").touch()
        (tmp_path / "clip").mkdir()

        answers = iter(["y", "q"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))

        with caplog.at_level(logging.WARNING):
            interactive_wizard(str(tmp_path))

        assert "Skipping loose video" in caplog.text
        assert "'clip.mp4'" in caplog.text
        assert "'clip' already exists" in caplog.text

    def test_organization_generic_exception(self, tmp_path, monkeypatch, caplog):
        """
        Scenario: A generic system error occurs while trying to create a clip folder.
        Expected: Wizard logs the error and continues to the next task.
        """
        (tmp_path / "broken_clip.mp4").touch()

        def mock_explode(*args, **kwargs):
            raise RuntimeError("Unexpected System Failure")

        monkeypatch.setattr("os.makedirs", mock_explode)

        answers = iter(["y", "q"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))

        with caplog.at_level(logging.ERROR):
            interactive_wizard(str(tmp_path))

        assert "Failed to organize video 'broken_clip.mp4'" in caplog.text
        assert "Unexpected System Failure" in caplog.text

    def test_too_many_folders_summary(self, tmp_path, monkeypatch, capsys):
        """
        Scenario: More than 10 folders need organization.
        Expected: Wizard truncates the list to keep the UI clean.
        """
        for i in range(11):
            (tmp_path / f"shot_{i}").mkdir()

        answers = iter(["n", "q"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))

        interactive_wizard(str(tmp_path))

        out = capsys.readouterr().out
        assert "...and 11 others." in out

    def test_status_reports(self, tmp_path, monkeypatch, capsys):
        """
        Scenario: User targets a project root containing a mix of shot types (Ready, Masked, and Raw).
        Expected: Wizard accurately counts and lists each shot in the 'STATUS REPORT' block based on folder contents.
        """

        def create_valid_input(path):
            input_dir = path / "Input"
            input_dir.mkdir(parents=True)
            img = np.zeros((1, 1, 3), dtype=np.uint8)
            cv2.imwrite(str(input_dir / "frame_0001.png"), img)

        ready_dir = tmp_path / "shot_ready"
        create_valid_input(ready_dir)
        (ready_dir / "AlphaHint").mkdir()
        (ready_dir / "AlphaHint" / "alpha.mov").write_text("dummy")

        masked_dir = tmp_path / "shot_masked"
        create_valid_input(masked_dir)
        (masked_dir / "VideoMamaMaskHint").mkdir()
        (masked_dir / "VideoMamaMaskHint" / "mask.png").write_text("dummy")

        raw_dir = tmp_path / "shot_raw"
        create_valid_input(raw_dir)

        answers = iter(["n", "q"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))

        interactive_wizard(str(tmp_path))

        out = capsys.readouterr().out

        assert "READY (AlphaHint found): 1" in out
        assert "MASKED (VideoMamaMaskHint found): 1" in out
        assert "RAW (Input only):        1" in out

    def test_mask_hint_as_file_detection(self, tmp_path, monkeypatch, capsys):
        """
        Scenario: A shot folder contains no 'VideoMamaMaskHint' directory, but has a file named 'videomamamaskhint.mp4'.
        Expected: Wizard identifies the shot as MASKED by finding the file.
        """
        shot_dir = tmp_path / "shot_vfx"
        shot_dir.mkdir()
        (shot_dir / "Input").mkdir()
        (shot_dir / "videomamamaskhint.mp4").touch()

        monkeypatch.setattr("clip_manager.ClipEntry.find_assets", lambda _: None)

        answers = iter(["n", "q"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))

        interactive_wizard(str(tmp_path))

        out = capsys.readouterr().out
        assert "MASKED (VideoMamaMaskHint found): 1" in out

    def test_triggers_gvm_process(self, tmp_path, monkeypatch, capsys):
        """
        Scenario: User has 1 RAW clip and selects [g] to run GVM.
        Expected: generate_alphas is called with the 'raw' list containing our clip.
        """
        raw_dir = tmp_path / "shot_raw"
        (raw_dir / "Input").mkdir(parents=True)
        (raw_dir / "Input" / "video.mp4").touch()

        captured_args = []

        def mock_generate_alphas(clips, device=None):
            captured_args.append(clips)
            print("MOCK: GVM Pipeline Triggered")

        monkeypatch.setattr("corridorkey_cli.generate_alphas", mock_generate_alphas)
        answers = iter(["n", "g", "y", "", "q"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))

        interactive_wizard(str(tmp_path))

        out = capsys.readouterr().out
        assert "MOCK: GVM Pipeline Triggered" in out
        assert len(captured_args) == 1
        assert captured_args[0][0].name == "shot_raw"

    def test_gvm_confirmation_decline(self, tmp_path, monkeypatch):
        """
        Scenario: User selects GVM [g] but chooses 'n' at the 'Proceed?' prompt.
        Expected: Wizard skips generate_alphas and returns to the main action loop.
        """
        (tmp_path / "raw_shot").mkdir()
        monkeypatch.setattr("clip_manager.ClipEntry.find_assets", lambda _: None)
        call_tracker = {"called": False}

        def mock_gvm(clips, device=None):
            call_tracker["called"] = True

        monkeypatch.setattr("corridorkey_cli.generate_alphas", mock_gvm)

        answers = iter(["n", "g", "n", "q"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))

        interactive_wizard(str(tmp_path))

        assert call_tracker["called"] is False

    def test_triggers_videomama(self, tmp_path, monkeypatch, capsys):
        """
        Scenario: User targets a folder where a clip has a VideoMamaMaskHint but no AlphaHint.
        Expected: Wizard identifies clip as 'MASKED', displays the [v] action, and calls run_videomama when selected.
        """
        masked_dir = tmp_path / "shot_masked"
        (masked_dir / "Input").mkdir(parents=True)
        (masked_dir / "VideoMamaMaskHint").mkdir()
        (masked_dir / "VideoMamaMaskHint" / "mask.png").touch()
        captured_args = []

        def mock_videomama(clips, chunk_size=50, device=None):
            captured_args.append(clips)
            print("MOCK: VideoMaMa Triggered")

        monkeypatch.setattr("corridorkey_cli.run_videomama", mock_videomama)

        answers = iter(["n", "v", "", "q"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))

        interactive_wizard(str(tmp_path))

        out = capsys.readouterr().out
        assert "MOCK: VideoMaMa Triggered" in out
        assert captured_args[0][0].name == "shot_masked"

    def test_triggers_inference(self, tmp_path, monkeypatch):
        """
        Scenario: User targets a folder where a clip has an existing AlphaHint.
        Expected: Wizard identifies the clip as 'READY', displays the [i] action, and calls run_inference when selected.
        """
        ready_dir = tmp_path / "shot_ready"
        (ready_dir / "Input").mkdir(parents=True)
        (ready_dir / "AlphaHint").mkdir()

        from clip_manager import ClipAsset

        def mock_find_assets(self_entry):
            self_entry.input_asset = ClipAsset(str(ready_dir / "Input"), "video")
            self_entry.alpha_asset = ClipAsset(str(ready_dir / "AlphaHint"), "video")

            monkeypatch.setattr(self_entry.input_asset, "frame_count", 10)
            monkeypatch.setattr(self_entry.alpha_asset, "frame_count", 10)

        monkeypatch.setattr("clip_manager.ClipEntry.find_assets", mock_find_assets)

        captured_args = []
        monkeypatch.setattr("corridorkey_cli.run_inference", lambda clips, device=None: captured_args.append(clips))

        answers = iter(["n", "i", "", "q"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))

        interactive_wizard(str(tmp_path))

        assert len(captured_args) > 0, "run_inference was never called!"
        assert captured_args[0][0].name == "shot_ready"

    def test_inference_exception_handling(self, tmp_path, monkeypatch, caplog):
        """
        Scenario: run_inference raises a RuntimeError (e.g., GPU Out of Memory).
        Expected: Wizard logs the error but remains alive in the loop.
        """
        shot_dir = tmp_path / "ready_shot"
        shot_dir.mkdir()

        from clip_manager import ClipAsset

        def mock_find(self_entry):
            self_entry.input_asset = ClipAsset(str(shot_dir), "video")
            self_entry.alpha_asset = ClipAsset(str(shot_dir), "video")

        monkeypatch.setattr("clip_manager.ClipEntry.find_assets", mock_find)

        def mock_crash(clips, device=None):
            raise RuntimeError("CUDA Error: Out of Memory")

        monkeypatch.setattr("corridorkey_cli.run_inference", mock_crash)

        answers = iter(["n", "i", "", "q"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))

        with caplog.at_level(logging.ERROR):
            interactive_wizard(str(tmp_path))

        assert "Inference failed: CUDA Error: Out of Memory" in caplog.text

    def test_invalid_menu_selection_retry(self, tmp_path, monkeypatch, capsys):
        """
        Scenario: User enters an invalid menu option (e.g., 'z').
        Expected: Wizard prints 'Invalid selection.' and continues the loop (Lines 283-284).
        """
        (tmp_path / "shot").mkdir()
        monkeypatch.setattr("clip_manager.ClipEntry.find_assets", lambda _: None)

        answers = iter(["n", "z", "q"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))

        interactive_wizard(str(tmp_path))

        out = capsys.readouterr().out
        assert "Invalid selection." in out

    def test_rescan_loops(self, tmp_path, monkeypatch, capsys):
        """
        Scenario: User selects the [r] Re-Scan option from the main action menu.
        Expected: Wizard prints a 'Re-scanning...' message and re-enters the status/action loop without exiting.
        """
        shot_dir = tmp_path / "shot_a"
        (shot_dir / "Input").mkdir(parents=True)
        monkeypatch.setattr("clip_manager.ClipEntry.find_assets", lambda _: None)

        answers = iter(["n", "r", "q"])
        monkeypatch.setattr("builtins.input", lambda _: next(answers))

        interactive_wizard(str(tmp_path))

        out = capsys.readouterr().out
        assert "Re-scanning..." in out
        assert "Wizard Complete. Goodbye!" in out


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMainEntryPoint:
    def test_main_configures_and_runs_wizard(self, monkeypatch, capsys):
        """
        Scenario: Script is run via CLI: --action wizard --win_path ...
        Expected: main() sets up environment and calls interactive_wizard.
        """
        monkeypatch.setattr("sys.argv", ["corridorkey_cli.py", "--action", "wizard", "--win_path", "C:/Test"])
        monkeypatch.setattr(
            "corridorkey_cli.interactive_wizard", lambda path, device: print(f"RUNNING_WIZARD_AT_{path}")
        )

        main()

        out = capsys.readouterr().out
        assert "RUNNING_WIZARD_AT_C:/Test" in out

    def test_main_action_generate_alphas(self, monkeypatch):
        """
        Scenario: User runs 'python corridorkey_cli.py --action generate_alphas'.
        Expected: main() calls scan_clips() and passes the result to generate_alphas.
        """
        monkeypatch.setattr("sys.argv", ["corridorkey_cli.py", "--action", "generate_alphas"])
        dummy_clips = ["clip1", "clip2"]
        monkeypatch.setattr("corridorkey_cli.scan_clips", lambda: dummy_clips)
        call_tracker = {"called_with": None}

        def mock_gen(clips, device=None):
            call_tracker["called_with"] = clips

        monkeypatch.setattr("corridorkey_cli.generate_alphas", mock_gen)

        main()

        assert call_tracker["called_with"] == dummy_clips

    def test_main_action_run_inference(self, monkeypatch):
        """
        Scenario: User runs 'python corridorkey_cli.py --action run_inference'.
        Expected: main() calls scan_clips() and passes the result to run_inference.
        """
        monkeypatch.setattr("sys.argv", ["corridorkey_cli.py", "--action", "run_inference"])
        dummy_clips = ["shot_01", "shot_02"]
        monkeypatch.setattr("corridorkey_cli.scan_clips", lambda: dummy_clips)
        call_tracker = {"clips": None}

        def mock_inf(clips, device=None):
            call_tracker["clips"] = clips

        monkeypatch.setattr("corridorkey_cli.run_inference", mock_inf)

        main()

        assert call_tracker["clips"] == dummy_clips

    def test_main_wizard_missing_path(self, monkeypatch, capsys):
        """
        Scenario: User runs --action wizard but forgets to provide --win_path.
        Expected: main() prints a missing path error and terminates.
        """
        monkeypatch.setattr("sys.argv", ["corridorkey_cli.py", "--action", "wizard"])
        monkeypatch.setattr("corridorkey_cli.interactive_wizard", lambda path, device: None)

        main()

        out = capsys.readouterr().out
        assert "Error: --win_path required for wizard." in out

    def test_main_keyboard_interrupt(self, monkeypatch, capsys):
        """
        Scenario: User hits Ctrl+C during execution.
        Expected: Script catches KeyboardInterrupt and exits with code 130.
        """
        monkeypatch.setattr("sys.argv", ["corridorkey_cli.py", "--action", "list"])

        def mock_interrupt():
            raise KeyboardInterrupt

        monkeypatch.setattr("corridorkey_cli.scan_clips", mock_interrupt)

        with pytest.raises(SystemExit) as e:
            main()

        assert e.value.code == 130
        assert "Interrupted" in capsys.readouterr().out

    def test_main_generic_exception_exit(self, monkeypatch, caplog):
        """
        Scenario: A low-level RuntimeError occurs during a standard CLI action (e.g., 'list').
        Expected: main() catches the Exception, logs the error string, and exits with code 1.
        """
        monkeypatch.setattr("sys.argv", ["corridorkey_cli.py", "--action", "list"])
        error_msg = "Nuclear Meltodwn"

        def mock_explode():
            raise RuntimeError(error_msg)

        monkeypatch.setattr("corridorkey_cli.scan_clips", mock_explode)

        with pytest.raises(SystemExit) as e_info:
            with caplog.at_level("ERROR"):
                main()

        assert e_info.value.code == 1
        assert error_msg in caplog.text
