import os
from unittest.mock import MagicMock, patch
import numpy as np

# We import the objects we want to test
from clip_manager import run_inference, ClipEntry, ClipAsset, InferenceSettings

class TestClipManagerInference:
    """
    Test suite for the CorridorKey inference pipeline.
    
    Since this project relies heavily on PyTorch and Triton (which requires dedicated GPU
    hardware and takes a long time to load), we CANNOT run the real model in our tests.
    
    Instead, we use a testing strategy called "Mocking". 
    We will create fake (Mock) versions of:
      1. The Inference Engine (so we don't load the Neural Network)
      2. OpenCV Image Reading (so we don't need real video files on disk)
      3. OpenCV Image Writing (so we don't litter the disk with output files)
    
    This ensures our pipeline logic (loops, progress callbacks, directory setup) is
    tested instantly and reliably.
    """

    @patch("clip_manager.cv2.imread")
    @patch("clip_manager.cv2.cvtColor")
    def test_run_inference_basic_flow(self, mock_cvt_color, mock_imread, tmp_path):
        """
        Tests that run_inference correctly loops through frames and calls the engine.
        
        We use @patch decorators above the function. This tells Python:
        "Whenever clip_manager tries to use cv2.imread or cv2.cvtColor, intercept that 
        call and give me a MagicMock object instead."
        """
        
        # 1. Setup Fake Data
        # We need a temporary directory (tmp_path is provided by pytest) to act as our clip folder.
        clip_root = os.path.join(tmp_path, "TestClip")
        os.makedirs(os.path.join(clip_root, "Input"))
        os.makedirs(os.path.join(clip_root, "AlphaHint"))
        
        # We create fake files so the directory scanning logic finds "frames"
        with open(os.path.join(clip_root, "Input", "frame_000.png"), "w") as f: f.write("fake")
        with open(os.path.join(clip_root, "Input", "frame_001.png"), "w") as f: f.write("fake")
        with open(os.path.join(clip_root, "AlphaHint", "alpha_000.png"), "w") as f: f.write("fake")
        with open(os.path.join(clip_root, "AlphaHint", "alpha_001.png"), "w") as f: f.write("fake")

        # Create the Python object representing the clip structure
        clip = ClipEntry("TestClip", clip_root)
        clip.input_asset = ClipAsset(os.path.join(clip_root, "Input"), "sequence")
        clip.alpha_asset = ClipAsset(os.path.join(clip_root, "AlphaHint"), "sequence")
        
        # Manually set the frame counts for our test
        clip.input_asset.frame_count = 2
        clip.alpha_asset.frame_count = 2

        # 2. Configure our Mocks
        # We want our fake OpenCV to return dummy arrays instead of actually reading the file.
        mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_cvt_color.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # We create a fake "engine" to pass into the function.
        # This is called *Dependency Injection*. We bypass the heavy `create_engine()` call 
        # by passing our lightweight fake directly.
        mock_engine = MagicMock()
        mock_engine.forward.return_value = {
            "matte": np.zeros((10, 10), dtype=np.float32),
            "fg": np.zeros((10, 10, 3), dtype=np.float32)
        }
        
        # We also want to track if the progress callbacks are fired
        mock_on_clip_start = MagicMock()
        mock_on_frame_complete = MagicMock()
        
        settings = InferenceSettings()

        # 3. Execute the Function
        # Notice we pass `engine_override=mock_engine`.
        with patch("clip_manager.cv2.imwrite") as mock_imwrite:
            run_inference(
                clips=[clip],
                settings=settings,
                on_clip_start=mock_on_clip_start,
                on_frame_complete=mock_on_frame_complete,
                engine_override=mock_engine
            )

        # 4. Assert Expected Behavior
        # We expect the engine's `forward` pipeline to have been called twice (once per frame)
        assert mock_engine.forward.call_count == 2
        
        # We expect the clip start callback to fire with 2 frames total
        mock_on_clip_start.assert_called_once_with("TestClip", 2)
        
        # We expect our dummy output directories to have been automatically created
        assert os.path.exists(os.path.join(clip_root, "Output", "FG"))
        assert os.path.exists(os.path.join(clip_root, "Output", "Matte"))

    @patch("clip_manager.cv2.imread")
    @patch("clip_manager.cv2.cvtColor")
    def test_run_inference_start_frame(self, mock_cvt_color, mock_imread, tmp_path):
        """
        Tests our newly implemented `--start-frame` functionality.
        If we have 3 frames (0, 1, 2) but provide start_frame=2, it should only 
        process exactly 1 frame.
        """
        clip_root = os.path.join(tmp_path, "TestClipStartFrame")
        os.makedirs(os.path.join(clip_root, "Input"))
        os.makedirs(os.path.join(clip_root, "AlphaHint"))
        
        for i in range(3):
            with open(os.path.join(clip_root, "Input", f"frame_00{i}.png"), "w") as f: f.write("fake")
            with open(os.path.join(clip_root, "AlphaHint", f"alpha_00{i}.png"), "w") as f: f.write("fake")

        clip = ClipEntry("TestClipStartFrame", clip_root)
        clip.input_asset = ClipAsset(os.path.join(clip_root, "Input"), "sequence")
        clip.alpha_asset = ClipAsset(os.path.join(clip_root, "AlphaHint"), "sequence")
        clip.input_asset.frame_count = 3
        clip.alpha_asset.frame_count = 3

        mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_cvt_color.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        
        mock_engine = MagicMock()
        mock_engine.forward.return_value = {
            "matte": np.zeros((10, 10), dtype=np.float32),
            "fg": np.zeros((10, 10, 3), dtype=np.float32)
        }
        
        mock_on_clip_start = MagicMock()

        # Execute with start_frame=2
        with patch("clip_manager.cv2.imwrite"):
            run_inference(
                clips=[clip],
                start_frame=2,
                on_clip_start=mock_on_clip_start,
                engine_override=mock_engine
            )

        # Expected:
        # Total frames = 3. 
        # Range is (2, 3), meaning it only processes frame index 2 (1 total frame)
        assert mock_engine.forward.call_count == 1
        mock_on_clip_start.assert_called_once_with("TestClipStartFrame", 1)
