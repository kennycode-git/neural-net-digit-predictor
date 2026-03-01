from pydantic import BaseModel, Field


class ExplainRequest(BaseModel):
    pixels: list[list[float]] = Field(
        ...,
        description="28×28 float array, values in [0,1], inverted (stroke=1, bg=0)",
    )
    preprocess_version: str = Field(..., description="Must be '1'")
    target: int = Field(..., ge=0, le=9, description="Target digit class (0–9)")


class ExplainResponse(BaseModel):
    heatmap_png_base64: str
    method: str
    target: int
    model_version: str
    preprocess_version: str
    git_commit: str


class HealthResponse(BaseModel):
    status: str


class VersionResponse(BaseModel):
    model_version: str
    preprocess_version: str
    git_commit: str
