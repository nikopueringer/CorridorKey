import av
import numpy as np
from fractions import Fraction


class ProResWriter:
    """Writes RGBA frames to a ProRes 4444 .mov file with embedded alpha channel."""

    def __init__(self, path, width, height, frame_rate=24):
        self.container = av.open(path, mode='w')
        rate = Fraction(frame_rate).limit_denominator(10000)
        self.stream = self.container.add_stream('prores_ks', rate=rate)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = 'yuva444p10le'
        self.stream.options = {'profile': '4444'}  # ProRes 4444 profile

    def write_frame(self, rgba_float):
        """
        Write a single RGBA frame.

        Args:
            rgba_float: numpy array [H, W, 4] float32, values in 0.0-1.0 range.
                        RGB should be in display-referred (sRGB/Rec.709) space.
                        Alpha should be linear.
        """
        # Convert float 0-1 to uint16 0-65535 for 16-bit interleaved RGBA
        rgba_16 = (np.clip(rgba_float, 0.0, 1.0) * 65535).astype(np.uint16)
        frame = av.VideoFrame.from_ndarray(rgba_16, format='rgba64le')
        # Reformat to the stream's pixel format (yuva444p10le)
        frame = frame.reformat(format=self.stream.pix_fmt)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
