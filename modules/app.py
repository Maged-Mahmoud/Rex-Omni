#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import zipfile
from typing import List, Optional, Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import uvicorn
from contextlib import asynccontextmanager
import yaml


from rex_omni import RexOmniWrapper, RexOmniVisualize


TaskType = Literal["detection"]  # extend later if Rex-Omni supports more tasks


def load_config():
    with open(os.path.abspath("config/global_config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# ---- Model singleton (loaded once) ----
REX_MODEL: Optional[RexOmniWrapper] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global REX_MODEL
    # load model parameters from config not env
    model_path = CONFIG.get("rex_model_path", "IDEA-Research/Rex-Omni")
    backend = CONFIG.get("rex_backend", "vllm")  # or " vllm"

    # Keep these configurable via config
    max_tokens = CONFIG.get("rex_max_tokens", 4096)
    temperature = CONFIG.get("rex_temperature", 0.0)
    top_p = CONFIG.get("rex_top_p", 0.05)
    top_k = CONFIG.get("rex_top_k", 1)
    sample = CONFIG.get("rex_sample", False)
    repetition_penalty = CONFIG.get("rex_repetition_penalty", 1.05)

    kwargs = {
        "max_model_len": CONFIG.get("max_model_len", 4096),
        "gpu_memory_utilization": CONFIG.get("gpu_memory_utilization", 0.3),
        "tensor_parallel_size": CONFIG.get("tensor_parallel_size", 1),
    }

    REX_MODEL = RexOmniWrapper(
        model_path=model_path,
        backend=backend,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        **kwargs,
    )
    yield
    # Cleanup if needed

app = FastAPI(title="NMDC Image Service (Detection)", version="1.0.0", lifespan=lifespan)


def _parse_categories(categories: str) -> List[str]:
    """
    Accepts comma-separated categories: "man,woman,laptop"
    """
    cats = [c.strip() for c in (categories or "").split(",") if c.strip()]
    if not cats:
        raise HTTPException(status_code=422, detail="categories is required (comma-separated).")
    return cats


def _load_image(file: UploadFile) -> Image.Image:
    try:
        content = file.file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image '{file.filename}': {e}")
    finally:
        try:
            file.file.close()
        except Exception:
            pass


def _render_visualization(image: Image.Image, predictions, font_size: int, draw_width: int, show_labels: bool) -> Image.Image:
    """
    RexOmniVisualize can save to path, but we want an in-memory image.
    If RexOmniVisualize returns a PIL image internally in your version, adjust here.
    Fallback: save to buffer via a temp approach.
    """
    vis = RexOmniVisualize(
        image=image,
        predictions=predictions,
        font_size=font_size,
        draw_width=draw_width,
        show_labels=show_labels,
    )

    # Many versions support exporting/saving; we’ll save into a BytesIO by using a temporary buffer path-less approach:
    # If your RexOmniVisualize only supports .save(path), we can write to BytesIO by saving PIL image.
    # Assuming it exposes the rendered image as `vis.image` (common pattern); if not, replace with your library’s attribute.
    rendered = getattr(vis, "image", None)
    if rendered is None:
        # If your library doesn't expose a PIL image, you can save to a temp file and re-open.
        raise HTTPException(
            status_code=500,
            detail="Function Visualize does not expose a rendered PIL image. "
                   "Update _render_visualization() to match your version."
        )
    return rendered


@app.post("/image_detection/v1/process")
async def process_images(
    task: TaskType = Form("detection"),
    categories: str = Form(..., description="Comma-separated categories, e.g. man,woman,laptop"),
    images: List[UploadFile] = File(..., description="One or multiple images"),
    # visualization controls
    font_size: int = Form(20),
    draw_width: int = Form(5),
    show_labels: bool = Form(True),
    output_format: Literal["jpg", "png"] = Form("jpg"),
):
    if REX_MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    cats = _parse_categories(categories)
    if not images:
        raise HTTPException(status_code=422, detail="At least one image is required.")

    results_payload = []
    output_blobs = []

    for img_file in images:
        pil_img = _load_image(img_file)

        results = REX_MODEL.inference(images=pil_img, task=task, categories=cats)
        if not results or not isinstance(results, list):
            raise HTTPException(status_code=500, detail="Unexpected inference output.")

        r0 = results[0]
        if not r0.get("success"):
            err = r0.get("error", "Unknown error")
            raise HTTPException(status_code=500, detail=f"Inference failed for {img_file.filename}: {err}")

        predictions = r0.get("extracted_predictions", [])
        rendered_img = _render_visualization(
            image=pil_img,
            predictions=predictions,
            font_size=font_size,
            draw_width=draw_width,
            show_labels=show_labels,
        )

        buf = io.BytesIO()
        fmt = "PNG" if output_format.lower() == "png" else "JPEG"
        rendered_img.save(buf, format=fmt)
        buf.seek(0)

        base_name = os.path.splitext(img_file.filename or "image")[0]
        out_name = f"{base_name}_{task}.{output_format.lower()}"

        output_blobs.append((out_name, buf.read()))
        results_payload.append({"input": img_file.filename, "output": out_name, "predictions_count": len(predictions)})

    # If one image => return it directly
    if len(output_blobs) == 1:
        out_name, out_bytes = output_blobs[0]
        media_type = "image/png" if out_name.lower().endswith(".png") else "image/jpeg"
        return StreamingResponse(io.BytesIO(out_bytes), media_type=media_type, headers={"Content-Disposition": f'inline; filename="{out_name}"'})

    # If many => return zip
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in output_blobs:
            zf.writestr(name, data)

        # Optional: include a small JSON summary
        import json
        zf.writestr("summary.json", json.dumps(results_payload, indent=2))

    zip_buf.seek(0)
    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="detection_outputs.zip"'},
    )


@app.get("/image_detection/health")
def health():
    return {"status": "ok", "model_loaded": REX_MODEL is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8452,)