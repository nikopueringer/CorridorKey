"""Pydantic request/response models for the WebUI API."""

from __future__ import annotations

from pydantic import BaseModel, Field

# --- Clips ---


class ClipAssetSchema(BaseModel):
    path: str
    asset_type: str
    frame_count: int


class ClipSchema(BaseModel):
    name: str
    root_path: str
    state: str
    input_asset: ClipAssetSchema | None = None
    alpha_asset: ClipAssetSchema | None = None
    mask_asset: ClipAssetSchema | None = None
    frame_count: int = 0
    completed_frames: int = 0
    has_outputs: bool = False
    warnings: list[str] = []
    error_message: str | None = None


class ClipListResponse(BaseModel):
    clips: list[ClipSchema]
    clips_dir: str


# --- Jobs ---


class InferenceParamsSchema(BaseModel):
    input_is_linear: bool = False
    despill_strength: float = Field(1.0, ge=0.0, le=1.0)
    auto_despeckle: bool = True
    despeckle_size: int = Field(400, ge=1)
    refiner_scale: float = Field(1.0, ge=0.0)


class OutputConfigSchema(BaseModel):
    fg_enabled: bool = True
    fg_format: str = "exr"
    matte_enabled: bool = True
    matte_format: str = "exr"
    comp_enabled: bool = True
    comp_format: str = "png"
    processed_enabled: bool = True
    processed_format: str = "exr"


class ExtractJobRequest(BaseModel):
    clip_names: list[str]


class PipelineJobRequest(BaseModel):
    """Full pipeline: extract (if needed) → GVM alpha → inference."""

    clip_names: list[str]
    alpha_method: str = "gvm"  # "gvm" or "videomama"
    params: InferenceParamsSchema = InferenceParamsSchema()
    output_config: OutputConfigSchema = OutputConfigSchema()


class InferenceJobRequest(BaseModel):
    clip_names: list[str]
    params: InferenceParamsSchema = InferenceParamsSchema()
    output_config: OutputConfigSchema = OutputConfigSchema()
    frame_range: tuple[int, int] | None = None


class GVMJobRequest(BaseModel):
    clip_names: list[str]


class VideoMaMaJobRequest(BaseModel):
    clip_names: list[str]
    chunk_size: int = 50


class JobSchema(BaseModel):
    id: str
    job_type: str
    clip_name: str
    status: str
    current_frame: int = 0
    total_frames: int = 0
    error_message: str | None = None


class JobListResponse(BaseModel):
    current: JobSchema | None = None
    queued: list[JobSchema] = []
    history: list[JobSchema] = []


# --- System ---


class DeviceResponse(BaseModel):
    device: str


class VRAMResponse(BaseModel):
    total: float = 0.0
    reserved: float = 0.0
    allocated: float = 0.0
    free: float = 0.0
    name: str = ""
    available: bool = False


# --- WebSocket ---


class WSMessage(BaseModel):
    type: str
    data: dict
