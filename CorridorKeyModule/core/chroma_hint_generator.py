"""
Adaptive chroma-key based alpha hint generator for CorridorKey.

Generates soft foreground/background masks from green-screen footage using
HSV color space analysis. Instead of hardcoding a generic "green" range,
this samples the actual green screen color from the footage and builds a
tight detection model around it.

No AI model required. Works on any subject type (people, props, VFX elements).
Instant — runs on any hardware.
"""
import os
import cv2
import numpy as np


def sample_screen_color(frame_bgr):
    """Detect the dominant green-screen color from a single frame.

    Strategy: Two-pass approach.
    1. Broad HSV pass finds anything vaguely green with decent saturation.
    2. Compute median H/S/V of those pixels, then do a TIGHT second pass
       around that median to isolate the core green-screen cluster.
    3. Build the final detection range from the core cluster's distribution.

    This adapts to the specific green screen — bright neon, dark fabric,
    unevenly lit, etc. — while avoiding shadow/edge pixels that would
    make the range too permissive.

    Args:
        frame_bgr: BGR uint8 numpy array [H, W, 3].

    Returns:
        Dict with HSV range, or None if no green screen detected.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Pass 1: Find anything greenish with reasonable saturation + brightness.
    # Higher S/V floor than before — we want the well-lit green, not shadows.
    broad_mask = cv2.inRange(hsv, (25, 50, 50), (95, 255, 255))

    green_ratio = np.count_nonzero(broad_mask) / broad_mask.size
    if green_ratio < 0.1:
        return None

    # Get median of the broad green pixels
    green_pixels = hsv[broad_mask > 0]
    h_med = np.median(green_pixels[:, 0])
    s_med = np.median(green_pixels[:, 1])
    v_med = np.median(green_pixels[:, 2])

    # Pass 2: Tight window around the median to get the core cluster.
    # This excludes shadow-green, spill-green, and edge pixels.
    core_mask = cv2.inRange(hsv,
                            (max(0, h_med - 15), max(0, s_med - 40), max(0, v_med - 50)),
                            (min(180, h_med + 15), min(255, s_med + 40), min(255, v_med + 50)))
    core_pixels = hsv[core_mask > 0]

    if len(core_pixels) < 1000:
        return None

    h_vals = core_pixels[:, 0].astype(np.float32)
    s_vals = core_pixels[:, 1].astype(np.float32)
    v_vals = core_pixels[:, 2].astype(np.float32)

    # Build detection range: adaptive hue, but keep S/V floors reasonable
    # so we don't eat into dark clothing or desaturated areas.
    h_pad = 10
    s_pad = 15
    v_pad = 20
    color_profile = {
        'h_low': max(0, np.percentile(h_vals, 5) - h_pad),
        'h_high': min(180, np.percentile(h_vals, 95) + h_pad),
        's_low': max(30, np.percentile(s_vals, 5) - s_pad),   # floor at 30
        's_high': min(255, np.percentile(s_vals, 95) + s_pad),
        'v_low': max(30, np.percentile(v_vals, 5) - v_pad),   # floor at 30
        'v_high': min(255, np.percentile(v_vals, 95) + v_pad),
        'h_median': float(np.median(h_vals)),
        's_median': float(np.median(s_vals)),
        'v_median': float(np.median(v_vals)),
        'green_coverage': float(np.count_nonzero(core_mask) / core_mask.size),
    }
    return color_profile


def generate_hint(frame_bgr, color_profile=None, blur_size=15):
    """Generate an alpha hint mask for a single BGR frame.

    Uses HSV color space to detect green-screen areas and marks everything
    else as foreground. If a color_profile is provided (from sample_screen_color),
    uses the tight adaptive range. Otherwise falls back to a broad default.

    Args:
        frame_bgr: BGR uint8 numpy array [H, W, 3].
        color_profile: Optional dict from sample_screen_color().
        blur_size: Gaussian blur kernel size for edge softening.

    Returns:
        numpy uint8 array [H, W] with values 0 (background/green) to
        255 (foreground/subject).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    if color_profile:
        lower = (int(color_profile['h_low']),
                 int(color_profile['s_low']),
                 int(color_profile['v_low']))
        upper = (int(color_profile['h_high']),
                 int(color_profile['s_high']),
                 int(color_profile['v_high']))
    else:
        # Default broad green range
        lower = (35, 40, 40)
        upper = (85, 255, 255)

    green_mask = cv2.inRange(hsv, lower, upper)

    # Foreground = NOT green
    mask = 255 - green_mask

    # Soften edges for smooth gradient transitions
    if blur_size > 0:
        k = blur_size if blur_size % 2 == 1 else blur_size + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def generate_hints_for_video(video_path, output_dir, max_frames=None):
    """Generate alpha hints for all frames in a video file.

    Samples the green-screen color from the first frame, then uses that
    tight color profile for all subsequent frames.

    Args:
        video_path: Path to input video.
        output_dir: Directory to write hint PNGs.
        max_frames: Optional limit on number of frames to process.

    Returns:
        Tuple of (frame_count, fps).
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    color_profile = None
    count = 0
    while count < total:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample green-screen color from first frame
        if count == 0:
            color_profile = sample_screen_color(frame)
            if color_profile:
                print(f"  Screen color detected: H={color_profile['h_median']:.0f} "
                      f"S={color_profile['s_median']:.0f} V={color_profile['v_median']:.0f} "
                      f"(coverage: {color_profile['green_coverage']:.0%})")
                print(f"  Adaptive range: H[{color_profile['h_low']:.0f}-{color_profile['h_high']:.0f}] "
                      f"S[{color_profile['s_low']:.0f}-{color_profile['s_high']:.0f}] "
                      f"V[{color_profile['v_low']:.0f}-{color_profile['v_high']:.0f}]")
            else:
                print("  No dominant green detected — using default broad range")

        mask = generate_hint(frame, color_profile=color_profile)
        cv2.imwrite(os.path.join(output_dir, f"{count:05d}.png"), mask)
        count += 1
        if (count % 100 == 0) or count == 1:
            print(f"  Chroma hints: {count}/{total}")

    cap.release()
    return count, fps
