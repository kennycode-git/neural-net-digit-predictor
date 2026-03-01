"""
FastAPI backend for explainability heatmaps.

Endpoints:
  GET  /v1/health
  GET  /v1/version
  POST /v1/explain?method=saliency|gradcam

Rate limit: 10 explain requests / minute / IP (slowapi).
"""
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .schemas import ExplainRequest, ExplainResponse, HealthResponse, VersionResponse
from .version import GIT_COMMIT, MODEL_VERSION, PREPROCESS_VERSION

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Digit-Net Backend", version=MODEL_VERSION)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allow the Next.js dev server and any custom domain
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/v1/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@app.get("/v1/version", response_model=VersionResponse)
async def version():
    return VersionResponse(
        model_version=MODEL_VERSION,
        preprocess_version=PREPROCESS_VERSION,
        git_commit=GIT_COMMIT,
    )


@app.post("/v1/explain", response_model=ExplainResponse)
@limiter.limit("10/minute")
async def explain(
    request: Request,
    body: ExplainRequest,
    method: str = "saliency",
):
    from .explain import gradcam_heatmap, saliency_heatmap

    if body.preprocess_version != PREPROCESS_VERSION:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"preprocess_version mismatch: expected '{PREPROCESS_VERSION}', "
                   f"got '{body.preprocess_version}'",
        )

    if len(body.pixels) != 28 or any(len(row) != 28 for row in body.pixels):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="pixels must be 28×28")

    if method == "gradcam":
        heatmap_b64 = gradcam_heatmap(body.pixels, body.target)
    else:
        heatmap_b64 = saliency_heatmap(body.pixels, body.target)

    return ExplainResponse(
        heatmap_png_base64=heatmap_b64,
        method=method,
        target=body.target,
        model_version=MODEL_VERSION,
        preprocess_version=PREPROCESS_VERSION,
        git_commit=GIT_COMMIT,
    )
