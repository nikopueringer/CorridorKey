"""Clip entry data model and state machine.

State machine transitions:

    EXTRACTING -> RAW        extraction completes
    EXTRACTING -> ERROR      extraction fails
    RAW        -> MASKED     user provides a VideoMaMa mask
    RAW        -> READY      alpha generator produces AlphaHint
    RAW        -> ERROR      alpha generation or scan fails
    MASKED     -> READY      VideoMaMa generates alpha from mask
    MASKED     -> ERROR      VideoMaMa fails
    READY      -> COMPLETE   inference succeeds
    READY      -> ERROR      inference fails
    ERROR      -> RAW        retry from scratch
    ERROR      -> MASKED     retry with mask
    ERROR      -> READY      retry inference only
    ERROR      -> EXTRACTING retry extraction
    COMPLETE   -> READY      reprocess with different params
"""

from __future__ import annotations

import glob as glob_module
import logging
import os
from dataclasses import dataclass, field
from enum import Enum

from corridorkey.errors import ClipScanError, InvalidStateTransitionError
from corridorkey.models import InOutRange
from corridorkey.natural_sort import natsorted
from corridorkey.project import is_image_file as _is_image_file
from corridorkey.project import is_video_file as _is_video_file

logger = logging.getLogger(__name__)


class ClipState(Enum):
    """Processing state of a single clip."""

    EXTRACTING = "EXTRACTING"
    RAW = "RAW"
    MASKED = "MASKED"
    READY = "READY"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


# Valid state transitions: from_state -> set of allowed to_states.
_TRANSITIONS: dict[ClipState, set[ClipState]] = {
    ClipState.EXTRACTING: {ClipState.RAW, ClipState.ERROR},
    ClipState.RAW: {ClipState.MASKED, ClipState.READY, ClipState.ERROR},
    ClipState.MASKED: {ClipState.READY, ClipState.ERROR},
    ClipState.READY: {ClipState.COMPLETE, ClipState.ERROR},
    ClipState.COMPLETE: {ClipState.READY},
    ClipState.ERROR: {ClipState.RAW, ClipState.MASKED, ClipState.READY, ClipState.EXTRACTING},
}


@dataclass
class ClipAsset:
    """An input source - either an image sequence directory or a video file.

    Attributes:
        path: Absolute path to the directory or video file.
        asset_type: Either 'sequence' or 'video'.
        frame_count: Number of frames detected at construction time.
    """

    path: str
    asset_type: str
    frame_count: int = 0

    def __post_init__(self) -> None:
        self._calculate_length()

    def _calculate_length(self) -> None:
        """Populate frame_count by inspecting the asset on disk."""
        if self.asset_type == "sequence":
            if os.path.isdir(self.path):
                files = [f for f in os.listdir(self.path) if _is_image_file(f)]
                self.frame_count = len(files)
            else:
                self.frame_count = 0
        elif self.asset_type == "video":
            try:
                import cv2

                cap = cv2.VideoCapture(self.path)
                if cap.isOpened():
                    self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if self.frame_count == 0:
                        logger.warning("Video reports 0 frames, file may be corrupted: %s", self.path)
                cap.release()
            except Exception as e:
                logger.debug("Video frame count detection failed for %s: %s", self.path, e)
                self.frame_count = 0

    def get_frame_files(self) -> list[str]:
        """Return naturally sorted frame filenames for sequence assets.

        Uses natural sort so frame_2 sorts before frame_10.

        Returns:
            List of filenames (not full paths). Empty list for video assets.
        """
        if self.asset_type != "sequence" or not os.path.isdir(self.path):
            return []
        return natsorted([f for f in os.listdir(self.path) if _is_image_file(f)])


@dataclass
class ClipEntry:
    """A single shot/clip with its assets and processing state.

    Attributes:
        name: Human-readable clip name (may be overridden by display_name in JSON).
        root_path: Absolute path to the clip folder.
        state: Current processing state.
        input_asset: Input image sequence or video.
        alpha_asset: AlphaHint image sequence or video.
        mask_asset: Optional VideoMaMa mask hint.
        in_out_range: Per-clip in/out markers. None means process the full clip.
        warnings: Non-fatal warnings accumulated during scanning.
        error_message: Set when state is ERROR.
        extraction_progress: Progress fraction (0.0-1.0) during EXTRACTING.
        extraction_total: Total frames expected during extraction.
        _processing: Internal lock; True while a GPU job is active on this clip.
    """

    name: str
    root_path: str
    state: ClipState = ClipState.RAW
    input_asset: ClipAsset | None = None
    alpha_asset: ClipAsset | None = None
    mask_asset: ClipAsset | None = None
    in_out_range: InOutRange | None = None
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None
    extraction_progress: float = 0.0
    extraction_total: int = 0
    # True while a GPU job is actively working on this clip; watcher skips reclassification.
    _processing: bool = field(default=False, repr=False)

    @property
    def is_processing(self) -> bool:
        """True while a GPU job is actively working on this clip."""
        return self._processing

    def set_processing(self, value: bool) -> None:
        """Set the processing lock. Watcher skips reclassification while True.

        Args:
            value: True to lock, False to release.
        """
        self._processing = value

    def transition_to(self, new_state: ClipState) -> None:
        """Attempt a state transition.

        Args:
            new_state: Target state.

        Raises:
            InvalidStateTransitionError: If the transition is not allowed.
        """
        if new_state not in _TRANSITIONS.get(self.state, set()):
            raise InvalidStateTransitionError(self.name, self.state.value, new_state.value)
        old = self.state
        self.state = new_state
        if new_state != ClipState.ERROR:
            self.error_message = None
        logger.debug("Clip '%s': %s -> %s", self.name, old.value, new_state.value)

    def set_error(self, message: str) -> None:
        """Transition to ERROR state and record a message.

        Args:
            message: Description of the failure.
        """
        self.transition_to(ClipState.ERROR)
        self.error_message = message

    @property
    def output_dir(self) -> str:
        """Absolute path to the Output subdirectory."""
        return os.path.join(self.root_path, "Output")

    @property
    def has_outputs(self) -> bool:
        """True if the Output directory exists and contains at least one file."""
        out = self.output_dir
        if not os.path.isdir(out):
            return False
        for subdir in ("FG", "Matte", "Comp", "Processed"):
            d = os.path.join(out, subdir)
            if os.path.isdir(d) and os.listdir(d):
                return True
        return False

    def completed_frame_count(self) -> int:
        """Count output frames that have all enabled outputs written.

        Returns:
            Number of fully-completed frames.
        """
        return len(self.completed_stems())

    def completed_stems(self) -> set[str]:
        """Return frame stems that have all enabled outputs complete.

        Reads the run manifest to determine which outputs to check.
        Falls back to FG+Matte intersection when no manifest exists.

        Returns:
            Set of stem strings (e.g. {'frame_000001', 'frame_000002'}).
        """
        manifest = self._read_manifest()
        enabled = manifest.get("enabled_outputs", []) if manifest else ["fg", "matte"]

        dir_map = {
            "fg": os.path.join(self.output_dir, "FG"),
            "matte": os.path.join(self.output_dir, "Matte"),
            "comp": os.path.join(self.output_dir, "Comp"),
            "processed": os.path.join(self.output_dir, "Processed"),
        }

        stem_sets: list[set[str]] = []
        for output_name in enabled:
            d = dir_map.get(output_name)
            if d and os.path.isdir(d):
                stems = {os.path.splitext(f)[0] for f in os.listdir(d) if _is_image_file(f)}
                stem_sets.append(stems)
            else:
                return set()

        if not stem_sets:
            return set()

        result = stem_sets[0]
        for s in stem_sets[1:]:
            result &= s
        return result

    def _read_manifest(self) -> dict | None:
        """Read the run manifest if it exists.

        Returns:
            Parsed manifest dict, or None if not found or unreadable.
        """
        manifest_path = os.path.join(self.output_dir, ".corridorkey_manifest.json")
        if not os.path.isfile(manifest_path):
            return None
        try:
            import json

            with open(manifest_path) as f:
                return json.load(f)
        except Exception as e:
            logger.debug("Failed to read manifest at %s: %s", manifest_path, e)
            return None

    def _resolve_original_path(self) -> str | None:
        """Resolve the original video path from clip.json or project.json.

        Returns:
            Absolute path to the original video, or None if not found.
        """
        from corridorkey.project import read_clip_json, read_project_json

        data = read_clip_json(self.root_path) or read_project_json(self.root_path)
        if not data:
            return None
        source = data.get("source", {})
        path = source.get("original_path")
        if path and os.path.isfile(path):
            return path
        return None

    def find_assets(self) -> None:
        """Scan the clip directory for Input, AlphaHint, and mask assets.

        Supports both the current format (Frames/, Source/) and the legacy
        format (Input/, Input.*) for backward compatibility. Updates state
        based on what is found.

        Raises:
            ClipScanError: If no valid input asset can be located.
        """
        frames_dir = os.path.join(self.root_path, "Frames")
        input_dir = os.path.join(self.root_path, "Input")
        source_dir = os.path.join(self.root_path, "Source")

        if os.path.isdir(frames_dir) and os.listdir(frames_dir):
            self.input_asset = ClipAsset(frames_dir, "sequence")
        elif os.path.isdir(input_dir) and os.listdir(input_dir):
            self.input_asset = ClipAsset(input_dir, "sequence")
        elif os.path.isdir(source_dir):
            videos = [f for f in os.listdir(source_dir) if _is_video_file(f)]
            if videos:
                self.input_asset = ClipAsset(os.path.join(source_dir, videos[0]), "video")
            else:
                original = self._resolve_original_path()
                if original:
                    self.input_asset = ClipAsset(original, "video")
                else:
                    raise ClipScanError(f"Clip '{self.name}': 'Source' dir has no video.")
        else:
            candidates = glob_module.glob(os.path.join(self.root_path, "[Ii]nput.*"))
            candidates = [c for c in candidates if _is_video_file(c)]
            if candidates:
                self.input_asset = ClipAsset(candidates[0], "video")
            elif os.path.isdir(input_dir):
                raise ClipScanError(f"Clip '{self.name}': Input dir is empty - no image files.")
            else:
                raise ClipScanError(f"Clip '{self.name}': no Input found.")

        from corridorkey.project import get_display_name

        display = get_display_name(self.root_path)
        if display != os.path.basename(self.root_path):
            self.name = display

        alpha_dir = os.path.join(self.root_path, "AlphaHint")
        if os.path.isdir(alpha_dir) and os.listdir(alpha_dir):
            self.alpha_asset = ClipAsset(alpha_dir, "sequence")

        mask_dir = os.path.join(self.root_path, "VideoMamaMaskHint")
        if os.path.isdir(mask_dir) and os.listdir(mask_dir):
            self.mask_asset = ClipAsset(mask_dir, "sequence")
        else:
            mask_candidates = glob_module.glob(os.path.join(self.root_path, "VideoMamaMaskHint.*"))
            mask_candidates = [c for c in mask_candidates if _is_video_file(c)]
            if mask_candidates:
                self.mask_asset = ClipAsset(mask_candidates[0], "video")

        from corridorkey.project import load_in_out_range

        self.in_out_range = load_in_out_range(self.root_path)

        self._resolve_state()

    def _resolve_state(self) -> None:
        """Set state based on assets present on disk.

        Recovers the furthest pipeline stage so the user never loses
        completed work after a restart or crash.

        Priority (highest first):
          COMPLETE   - all input frames have matching outputs (manifest-aware)
          READY      - AlphaHint covers all input frames
          MASKED     - VideoMaMa mask hint exists
          EXTRACTING - video source exists but no frame sequence yet
          RAW        - frame sequence exists, no alpha/mask/output
        """
        if self.alpha_asset is not None and self.input_asset is not None:
            completed = self.completed_stems()
            if completed and len(completed) >= self.input_asset.frame_count:
                self.state = ClipState.COMPLETE
                return

        if self.alpha_asset is not None:
            if self.input_asset is not None and self.alpha_asset.frame_count < self.input_asset.frame_count:
                logger.info(
                    "Clip '%s': partial alpha (%d/%d), staying at lower state",
                    self.name,
                    self.alpha_asset.frame_count,
                    self.input_asset.frame_count,
                )
            else:
                self.state = ClipState.READY
                return

        if self.mask_asset is not None:
            self.state = ClipState.MASKED
        elif self.input_asset is not None and self.input_asset.asset_type == "video":
            self.state = ClipState.EXTRACTING
        else:
            self.state = ClipState.RAW


def scan_project_clips(project_dir: str) -> list[ClipEntry]:
    """Scan a single project directory for its clips.

    v2 projects (with a clips/ subdir): each subdirectory inside clips/ is a clip.
    v1 projects (no clips/ subdir): the project dir itself is a single clip.

    Args:
        project_dir: Absolute path to a project folder.

    Returns:
        List of ClipEntry objects with root_path pointing to clip subdirectories.
    """
    from corridorkey.project import is_v2_project

    if is_v2_project(project_dir):
        clips_dir = os.path.join(project_dir, "clips")
        entries: list[ClipEntry] = []
        for item in sorted(os.listdir(clips_dir)):
            item_path = os.path.join(clips_dir, item)
            if item.startswith(".") or item.startswith("_"):
                continue
            if not os.path.isdir(item_path):
                continue
            clip = ClipEntry(name=item, root_path=item_path)
            try:
                clip.find_assets()
                entries.append(clip)
            except ClipScanError as e:
                logger.debug("%s", e)
        logger.info("Scanned v2 project %s: %d clip(s)", project_dir, len(entries))
        return entries

    clip = ClipEntry(name=os.path.basename(project_dir), root_path=project_dir)
    try:
        clip.find_assets()
        return [clip]
    except ClipScanError as e:
        logger.debug("%s", e)
        return []


def _looks_like_clip(path: str) -> bool:
    """Return True if *path* itself appears to be a clip root.

    A directory is treated as a clip root when it contains at least one of
    the recognised input subdirectories (``Frames``, ``Input``, ``Source``).

    Args:
        path: Absolute path to a directory.

    Returns:
        True when the directory looks like a clip root.
    """
    for subdir in ("Frames", "Input", "Source"):
        candidate = os.path.join(path, subdir)
        if os.path.isdir(candidate):
            return True
    return False


def scan_clips_dir(
    clips_dir: str,
    allow_standalone_videos: bool = True,
) -> list[ClipEntry]:
    """Scan a directory for clip folders and optionally standalone video files.

    For the Projects root: iterates project subdirectories and delegates to
    scan_project_clips() for each, flattening results.

    For non-Projects directories: scans subdirectories directly as clips
    (legacy behaviour for drag-and-dropped folders).

    If the directory itself looks like a single clip root (contains ``Input/``,
    ``Frames/``, or ``Source/`` directly), it is treated as a single clip
    rather than a container of clips.  This handles the common case where a
    user drags the clip folder itself onto the launcher instead of its parent.

    Folders without valid input assets are silently skipped.

    Args:
        clips_dir: Path to scan.
        allow_standalone_videos: When False, loose video files at the top
            level are ignored. Set False for the Projects root where videos
            live inside Source/ subdirectories.

    Returns:
        List of ClipEntry objects found in the directory.
    """
    entries: list[ClipEntry] = []
    if not os.path.isdir(clips_dir):
        logger.warning("Clips directory not found: %s", clips_dir)
        return entries

    from corridorkey.project import is_v2_project

    if is_v2_project(clips_dir):
        return scan_project_clips(clips_dir)

    # If the folder itself is a clip root, treat it as a single clip.
    if _looks_like_clip(clips_dir):
        name = os.path.basename(clips_dir)
        clip = ClipEntry(name=name, root_path=clips_dir)
        try:
            clip.find_assets()
            logger.info(
                "Dropped folder '%s' is itself a clip root; treating as single clip.",
                clips_dir,
            )
            return [clip]
        except ClipScanError as e:
            logger.debug("%s", e)
            return []

    seen_names: set[str] = set()

    for item in sorted(os.listdir(clips_dir)):
        item_path = os.path.join(clips_dir, item)

        if item.startswith(".") or item.startswith("_"):
            continue

        if os.path.isdir(item_path):
            from corridorkey.project import is_v2_project as _is_v2

            if _is_v2(item_path):
                for clip in scan_project_clips(item_path):
                    if clip.name not in seen_names:
                        entries.append(clip)
                        seen_names.add(clip.name)
            else:
                clip = ClipEntry(name=item, root_path=item_path)
                try:
                    clip.find_assets()
                    entries.append(clip)
                    seen_names.add(clip.name)
                except ClipScanError as e:
                    logger.debug("%s", e)

        elif allow_standalone_videos and os.path.isfile(item_path) and _is_video_file(item_path):
            stem = os.path.splitext(item)[0]
            if stem in seen_names:
                continue
            clip = ClipEntry(name=stem, root_path=clips_dir)
            clip.input_asset = ClipAsset(item_path, "video")
            clip.state = ClipState.EXTRACTING
            entries.append(clip)
            seen_names.add(stem)

    logger.info("Scanned %s: %d clip(s) found", clips_dir, len(entries))
    return entries
