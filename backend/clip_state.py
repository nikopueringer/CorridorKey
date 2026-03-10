"""Clip entry data model and state machine.

State Machine:
    EXTRACTING — Video input being extracted to image sequence
    RAW        — Input asset found, no alpha hint yet
    MASKED     — User mask provided (for VideoMaMa workflow)
    READY      — Alpha hint available (from GVM or VideoMaMa), ready for inference
    COMPLETE   — Inference outputs written
    ERROR      — Processing failed (can retry)

Transitions:
    EXTRACTING → RAW   (extraction completes)
    EXTRACTING → ERROR (extraction fails)
    RAW → MASKED       (user provides VideoMaMa mask)
    RAW → READY        (GVM auto-generates alpha)
    RAW → ERROR        (GVM/scan fails)
    MASKED → READY     (VideoMaMa generates alpha from user mask)
    MASKED → ERROR     (VideoMaMa fails)
    READY → COMPLETE   (inference succeeds)
    READY → ERROR      (inference fails)
    ERROR → RAW        (retry from scratch)
    ERROR → MASKED     (retry with mask)
    ERROR → READY      (retry inference)
    ERROR → EXTRACTING (retry extraction)
    COMPLETE → READY   (reprocess with different params)
"""

from __future__ import annotations

import glob as glob_module
import logging
import os
from dataclasses import dataclass, field
from enum import Enum

from .errors import ClipScanError, InvalidStateTransitionError
from .natural_sort import natsorted
from .project import is_image_file as _is_image_file
from .project import is_video_file as _is_video_file

logger = logging.getLogger(__name__)


class ClipState(Enum):
    EXTRACTING = "EXTRACTING"
    RAW = "RAW"
    MASKED = "MASKED"
    READY = "READY"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


# Valid transitions: from_state -> set of allowed to_states
_TRANSITIONS: dict[ClipState, set[ClipState]] = {
    ClipState.EXTRACTING: {ClipState.RAW, ClipState.ERROR},
    ClipState.RAW: {ClipState.MASKED, ClipState.READY, ClipState.ERROR},
    ClipState.MASKED: {ClipState.READY, ClipState.ERROR},
    ClipState.READY: {ClipState.COMPLETE, ClipState.ERROR},
    ClipState.COMPLETE: {ClipState.READY},  # reprocess with different params
    ClipState.ERROR: {ClipState.RAW, ClipState.MASKED, ClipState.READY, ClipState.EXTRACTING},
}


@dataclass
class ClipAsset:
    """Represents an input source — either an image sequence directory or a video file."""

    path: str
    asset_type: str  # 'sequence' or 'video'
    frame_count: int = 0

    def __post_init__(self):
        self._calculate_length()

    def _calculate_length(self):
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
                        logger.warning(f"Video reports 0 frames, file may be corrupted: {self.path}")
                cap.release()
            except Exception as e:
                logger.debug(f"Video frame count detection failed for {self.path}: {e}")
                self.frame_count = 0

    def get_frame_files(self) -> list[str]:
        """Return naturally sorted list of frame filenames for sequence assets.

        Uses natural sort so frame_2 sorts before frame_10 (not lexicographic).
        """
        if self.asset_type != "sequence" or not os.path.isdir(self.path):
            return []
        return natsorted([f for f in os.listdir(self.path) if _is_image_file(f)])


@dataclass
class InOutRange:
    """In/out frame range for sub-clip processing. Both indices inclusive, 0-based."""

    in_point: int
    out_point: int

    @property
    def frame_count(self) -> int:
        return self.out_point - self.in_point + 1

    def contains(self, index: int) -> bool:
        return self.in_point <= index <= self.out_point

    def to_dict(self) -> dict:
        return {"in_point": self.in_point, "out_point": self.out_point}

    @classmethod
    def from_dict(cls, d: dict) -> InOutRange:
        return cls(in_point=d["in_point"], out_point=d["out_point"])


@dataclass
class ClipEntry:
    """A single shot/clip with its assets and processing state."""

    name: str
    root_path: str
    state: ClipState = ClipState.RAW
    input_asset: ClipAsset | None = None
    alpha_asset: ClipAsset | None = None
    mask_asset: ClipAsset | None = None  # User-provided VideoMaMa mask
    in_out_range: InOutRange | None = None  # Per-clip in/out markers (None = full clip)
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None
    extraction_progress: float = 0.0  # 0.0 to 1.0 during EXTRACTING
    extraction_total: int = 0  # total frames expected during extraction
    _processing: bool = field(default=False, repr=False)  # lock: watcher must not reclassify

    @property
    def is_processing(self) -> bool:
        """True while a GPU job is actively working on this clip."""
        return self._processing

    def set_processing(self, value: bool) -> None:
        """Set processing lock. Watcher skips reclassification while True."""
        self._processing = value

    def transition_to(self, new_state: ClipState) -> None:
        """Attempt a state transition. Raises InvalidStateTransitionError if not allowed."""
        if new_state not in _TRANSITIONS.get(self.state, set()):
            raise InvalidStateTransitionError(self.name, self.state.value, new_state.value)
        old = self.state
        self.state = new_state
        if new_state != ClipState.ERROR:
            self.error_message = None
        logger.debug(f"Clip '{self.name}': {old.value} -> {new_state.value}")

    def set_error(self, message: str) -> None:
        """Transition to ERROR state with a message.

        Works from any state that allows ERROR transition
        (RAW, MASKED, READY — all can error now).
        """
        self.transition_to(ClipState.ERROR)
        self.error_message = message

    @property
    def output_dir(self) -> str:
        return os.path.join(self.root_path, "Output")

    @property
    def has_outputs(self) -> bool:
        """Check if output directory exists with content."""
        out = self.output_dir
        if not os.path.isdir(out):
            return False
        for subdir in ("FG", "Matte", "Comp", "Processed"):
            d = os.path.join(out, subdir)
            if os.path.isdir(d) and os.listdir(d):
                return True
        return False

    def completed_frame_count(self) -> int:
        """Count existing output frames for resume support.

        Manifest-aware: reads .corridorkey_manifest.json to determine which
        outputs were enabled. Falls back to FG+Matte intersection if no manifest.
        """
        return len(self.completed_stems())

    def completed_stems(self) -> set[str]:
        """Return set of frame stems that have all enabled outputs complete.

        Reads the run manifest to determine which outputs to check.
        Falls back to FG+Matte intersection if no manifest exists.
        """
        manifest = self._read_manifest()
        if manifest:
            enabled = manifest.get("enabled_outputs", [])
        else:
            enabled = ["fg", "matte"]

        dir_map = {
            "fg": os.path.join(self.output_dir, "FG"),
            "matte": os.path.join(self.output_dir, "Matte"),
            "comp": os.path.join(self.output_dir, "Comp"),
            "processed": os.path.join(self.output_dir, "Processed"),
        }

        stem_sets = []
        for output_name in enabled:
            d = dir_map.get(output_name)
            if d and os.path.isdir(d):
                stems = {os.path.splitext(f)[0] for f in os.listdir(d) if _is_image_file(f)}
                stem_sets.append(stems)
            else:
                # Required dir missing → no complete frames
                return set()

        if not stem_sets:
            return set()

        # Intersection: frame complete only if ALL enabled outputs exist
        result = stem_sets[0]
        for s in stem_sets[1:]:
            result &= s
        return result

    def _read_manifest(self) -> dict | None:
        """Read the run manifest if it exists."""
        manifest_path = os.path.join(self.output_dir, ".corridorkey_manifest.json")
        if not os.path.isfile(manifest_path):
            return None
        try:
            import json

            with open(manifest_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Failed to read manifest at {manifest_path}: {e}")
            return None

    def _resolve_original_path(self) -> str | None:
        """Resolve the original video path from clip.json or project.json."""
        from .project import _read_clip_or_project_json

        data = _read_clip_or_project_json(self.root_path)
        if not data:
            return None
        source = data.get("source", {})
        path = source.get("original_path")
        if path and os.path.isfile(path):
            return path
        return None

    def find_assets(self) -> None:
        """Scan the clip directory for Input, AlphaHint, and mask assets.

        Updates state accordingly. Supports both new format (Frames/, Source/)
        and legacy format (Input/, Input.*) for backward compatibility.
        """
        # Input asset — check new names first, fall back to legacy
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
                self.input_asset = ClipAsset(
                    os.path.join(source_dir, videos[0]),
                    "video",
                )
            else:
                # Source/ exists but is empty — check project.json for external reference
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
                raise ClipScanError(f"Clip '{self.name}': Input dir is empty — no image files.")
            else:
                raise ClipScanError(f"Clip '{self.name}': no Input found.")

        # Load display name from project.json if available
        from .project import get_display_name

        display = get_display_name(self.root_path)
        if display != os.path.basename(self.root_path):
            self.name = display

        # Alpha hint asset
        alpha_dir = os.path.join(self.root_path, "AlphaHint")
        if os.path.isdir(alpha_dir) and os.listdir(alpha_dir):
            self.alpha_asset = ClipAsset(alpha_dir, "sequence")

        # VideoMaMa mask hint — directory OR video file
        mask_dir = os.path.join(self.root_path, "VideoMamaMaskHint")
        if os.path.isdir(mask_dir) and os.listdir(mask_dir):
            self.mask_asset = ClipAsset(mask_dir, "sequence")
        else:
            # Check for mask video file (VideoMamaMaskHint.mp4 etc.)
            mask_candidates = glob_module.glob(os.path.join(self.root_path, "VideoMamaMaskHint.*"))
            mask_candidates = [c for c in mask_candidates if _is_video_file(c)]
            if mask_candidates:
                self.mask_asset = ClipAsset(mask_candidates[0], "video")

        # Load in/out range from project.json
        from .project import load_in_out_range

        self.in_out_range = load_in_out_range(self.root_path)

        # Determine initial state
        self._resolve_state()

    def _resolve_state(self) -> None:
        """Set state based on what assets are present on disk.

        Recovers the furthest pipeline stage from disk contents so the
        user never loses completed work after a restart or crash.

        Priority (highest first):
          COMPLETE  — all input frames have matching outputs (manifest-aware)
          READY     — AlphaHint exists (inference-ready)
          MASKED    — VideoMaMa mask hint exists
          EXTRACTING — video source exists but no frame sequence yet
          RAW       — frame sequence exists, no alpha/mask/output
        """
        # Check COMPLETE first: outputs exist and cover all input frames
        if self.alpha_asset is not None and self.input_asset is not None:
            completed = self.completed_stems()
            if completed and len(completed) >= self.input_asset.frame_count:
                self.state = ClipState.COMPLETE
                return

        # READY: AlphaHint must cover ALL input frames (not partial)
        if self.alpha_asset is not None:
            if self.input_asset is not None and self.alpha_asset.frame_count < self.input_asset.frame_count:
                # Partial alpha — don't promote to READY, fall through
                logger.info(
                    f"Clip '{self.name}': partial alpha "
                    f"({self.alpha_asset.frame_count}/{self.input_asset.frame_count}), "
                    f"staying at lower state"
                )
            else:
                self.state = ClipState.READY
                return

        if self.mask_asset is not None:
            self.state = ClipState.MASKED
        elif self.input_asset is not None and self.input_asset.asset_type == "video":
            # Video input needs extraction to image sequence
            self.state = ClipState.EXTRACTING
        else:
            self.state = ClipState.RAW


def scan_project_clips(project_dir: str) -> list[ClipEntry]:
    """Scan a single project directory for its clips.

    v2 projects (with ``clips/`` subdir): each subdirectory inside clips/ is a clip.
    v1 projects (no ``clips/`` subdir): the project dir itself is a single clip.

    Args:
        project_dir: Absolute path to a project folder.

    Returns:
        List of ClipEntry objects with root_path pointing to clip subdirectories.
    """
    from .project import is_v2_project

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
                logger.debug(str(e))
        logger.info(f"Scanned v2 project {project_dir}: {len(entries)} clip(s)")
        return entries

    # v1 fallback: project_dir is itself a single clip
    clip = ClipEntry(name=os.path.basename(project_dir), root_path=project_dir)
    try:
        clip.find_assets()
        return [clip]
    except ClipScanError as e:
        logger.debug(str(e))
        return []


def scan_clips_dir(
    clips_dir: str,
    allow_standalone_videos: bool = True,
) -> list[ClipEntry]:
    """Scan a directory for clip folders and optionally standalone video files.

    For the Projects root: iterates project subdirectories and delegates to
    scan_project_clips() for each, flattening results.

    For non-Projects directories: scans subdirectories directly as clips
    (legacy behavior for drag-and-dropped folders).

    Folders without valid input assets are skipped (not added as broken clips).

    Args:
        clips_dir: Path to scan.
        allow_standalone_videos: If False, loose video files at top level are ignored.
            Set False for the Projects root where videos live inside Source/ subdirs.
    """
    entries: list[ClipEntry] = []
    if not os.path.isdir(clips_dir):
        logger.warning(f"Clips directory not found: {clips_dir}")
        return entries

    # If the directory itself is a v2 project, scan its clips directly
    from .project import is_v2_project

    if is_v2_project(clips_dir):
        return scan_project_clips(clips_dir)

    seen_names: set[str] = set()

    for item in sorted(os.listdir(clips_dir)):
        item_path = os.path.join(clips_dir, item)

        # Skip hidden and special items
        if item.startswith(".") or item.startswith("_"):
            continue

        if os.path.isdir(item_path):
            # Check if this is a v2 project container (has clips/ subdir)
            from .project import is_v2_project

            if is_v2_project(item_path):
                # v2 project: scan its clips/ subdirectory
                for clip in scan_project_clips(item_path):
                    if clip.name not in seen_names:
                        entries.append(clip)
                        seen_names.add(clip.name)
            else:
                # Flat clip dir or v1 project
                clip = ClipEntry(name=item, root_path=item_path)
                try:
                    clip.find_assets()
                    entries.append(clip)
                    seen_names.add(clip.name)
                except ClipScanError as e:
                    # Skip folders without valid input assets
                    logger.debug(str(e))

        elif allow_standalone_videos and os.path.isfile(item_path) and _is_video_file(item_path):
            # Standalone video file → treat as a clip needing extraction
            stem = os.path.splitext(item)[0]
            if stem in seen_names:
                continue  # folder clip already exists with this name
            clip = ClipEntry(name=stem, root_path=clips_dir)
            clip.input_asset = ClipAsset(item_path, "video")
            clip.state = ClipState.EXTRACTING
            entries.append(clip)
            seen_names.add(stem)

    logger.info(f"Scanned {clips_dir}: {len(entries)} clip(s) found")
    return entries
