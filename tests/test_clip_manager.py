import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from clip_manager import ClipAsset, ClipEntry, InferenceSettings, run_inference


class TestClipManagerInference(unittest.TestCase):
    """
    Test suite for the CorridorKey inference pipeline using mocks.
    """

    @patch("clip_manager.cv2.imread")
    @patch("clip_manager.cv2.cvtColor")
    @patch("clip_manager.cv2.imwrite")
    @patch("clip_manager.is_image_file")
    @patch("clip_manager.os.listdir")
    def test_run_inference_basic_flow(
        self, mock_listdir, mock_is_image_file, mock_imwrite, mock_cvt_color, mock_imread
    ):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_path:
            clip_root = os.path.join(tmp_path, "TestClip")
            os.makedirs(os.path.join(clip_root, "Input"))
            os.makedirs(os.path.join(clip_root, "AlphaHint"))

            clip = ClipEntry("TestClip", clip_root)
            clip.input_asset = ClipAsset(os.path.join(clip_root, "Input"), "sequence")
            clip.alpha_asset = ClipAsset(os.path.join(clip_root, "AlphaHint"), "sequence")
            clip.input_asset.frame_count = 2
            clip.alpha_asset.frame_count = 2

            def side_effect_imread(path, flags=None):
                if "alpha" in path.lower():
                    return np.zeros((10, 10), dtype=np.uint8)
                return np.zeros((10, 10, 3), dtype=np.uint8)

            mock_imread.side_effect = side_effect_imread
            mock_cvt_color.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
            mock_is_image_file.return_value = True

            def side_effect_listdir(path):
                if "input" in path.lower():
                    return ["frame_000.png", "frame_001.png"]
                else:
                    return ["alpha_000.png", "alpha_001.png"]

            mock_listdir.side_effect = side_effect_listdir

            import clip_manager

            clip_manager.cv2.IMREAD_UNCHANGED = -1
            clip_manager.cv2.IMREAD_ANYDEPTH = 2
            clip_manager.cv2.COLOR_BGR2RGB = 4

            mock_engine = MagicMock()
            mock_engine.forward.return_value = {
                "matte": np.zeros((10, 10), dtype=np.float32),
                "fg": np.zeros((10, 10, 3), dtype=np.float32),
            }

            mock_on_clip_start = MagicMock()
            mock_on_frame_complete = MagicMock()
            settings = InferenceSettings()

            run_inference(
                clips=[clip],
                settings=settings,
                on_clip_start=mock_on_clip_start,
                on_frame_complete=mock_on_frame_complete,
                engine_override=mock_engine,
            )

            self.assertEqual(mock_engine.process_frame.call_count, 2)
            mock_on_clip_start.assert_called_once_with("TestClip", 2)
            self.assertTrue(os.path.exists(os.path.join(clip_root, "Output", "FG")))
            self.assertTrue(os.path.exists(os.path.join(clip_root, "Output", "Matte")))

    @patch("clip_manager.cv2.imread")
    @patch("clip_manager.cv2.cvtColor")
    @patch("clip_manager.cv2.imwrite")
    @patch("clip_manager.is_image_file")
    @patch("clip_manager.os.listdir")
    def test_run_inference_start_frame(
        self, mock_listdir, mock_is_image_file, mock_imwrite, mock_cvt_color, mock_imread
    ):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_path:
            clip_root = os.path.join(tmp_path, "TestClipStartFrame")
            os.makedirs(os.path.join(clip_root, "Input"))
            os.makedirs(os.path.join(clip_root, "AlphaHint"))

            clip = ClipEntry("TestClipStartFrame", clip_root)
            clip.input_asset = ClipAsset(os.path.join(clip_root, "Input"), "sequence")
            clip.alpha_asset = ClipAsset(os.path.join(clip_root, "AlphaHint"), "sequence")
            clip.input_asset.frame_count = 3
            clip.alpha_asset.frame_count = 3

            def side_effect_imread(path, flags=None):
                if "alpha" in path.lower():
                    return np.zeros((10, 10), dtype=np.uint8)
                return np.zeros((10, 10, 3), dtype=np.uint8)

            mock_imread.side_effect = side_effect_imread
            mock_cvt_color.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
            mock_is_image_file.return_value = True

            def side_effect_listdir(path):
                if "input" in path.lower():
                    return ["frame_000.png", "frame_001.png", "frame_002.png"]
                else:
                    return ["alpha_000.png", "alpha_001.png", "alpha_002.png"]

            mock_listdir.side_effect = side_effect_listdir

            import clip_manager

            clip_manager.cv2.IMREAD_UNCHANGED = -1
            clip_manager.cv2.IMREAD_ANYDEPTH = 2
            clip_manager.cv2.COLOR_BGR2RGB = 4

            mock_engine = MagicMock()
            mock_engine.forward.return_value = {
                "matte": np.zeros((10, 10), dtype=np.float32),
                "fg": np.zeros((10, 10, 3), dtype=np.float32),
            }

            mock_on_clip_start = MagicMock()

            run_inference(clips=[clip], start_frame=2, on_clip_start=mock_on_clip_start, engine_override=mock_engine)

            self.assertEqual(mock_engine.process_frame.call_count, 1)
            mock_on_clip_start.assert_called_once_with("TestClipStartFrame", 1)


if __name__ == "__main__":
    unittest.main()
