from __future__ import annotations

from pathlib import Path
from typing import Iterable

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.main import run_pipeline
from src.utils.config import InstagramAsset, get_settings
from src.utils.logger import get_logger


app = FastAPI(title="Freedom Content Generator", version="0.1.0")
logger = get_logger("WebServer")

base_path = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(base_path / "templates"))

settings = get_settings()
app.mount("/static", StaticFiles(directory=settings.output_dir), name="static")


class GenerateRequest(BaseModel):
    industry: str = Field(..., min_length=1, description="Industry name to target during generation.")


def _serialize_assets(assets: Iterable[InstagramAsset]) -> list[dict[str, str]]:
    output_root = get_settings().output_dir
    serialized: list[dict[str, str]] = []
    for asset in assets:
        try:
            relative_path = asset.image_path.relative_to(output_root).as_posix()
        except ValueError:
            relative_path = asset.image_path.name
        serialized.append(
            {
                "headline": asset.idea.headline,
                "caption": asset.caption,
                "image_url": f"/static/{relative_path}",
                "source_url": asset.idea.source_url,
            }
        )
    return serialized


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate")
async def generate(request_data: GenerateRequest) -> dict[str, object]:
    industry = request_data.industry.strip()
    if not industry:
        raise HTTPException(status_code=400, detail="Industry name must not be empty.")

    logger.info("Received generation request for industry '%s'", industry)
    try:
        assets = await run_in_threadpool(run_pipeline, industry)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Generation failed for industry '%s': %s", industry, exc)
        raise HTTPException(status_code=500, detail="Failed to generate assets.") from exc

    if not assets:
        return {"status": "empty", "message": "No assets were generated. Try another industry."}

    serialized = _serialize_assets(assets)
    logger.info("Successfully generated %d assets for '%s'", len(serialized), industry)
    return {"status": "ok", "results": serialized}

