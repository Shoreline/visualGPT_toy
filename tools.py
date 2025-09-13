# tools.py
from PIL import Image, ImageDraw
import numpy as np, cv2, uuid, os
from pydantic import BaseModel, Field

WORKDIR = "artifacts"; os.makedirs(WORKDIR, exist_ok=True)

class ToolSpec(BaseModel):
    name: str
    description: str
    args_schema: dict

def _save(img: Image.Image) -> str:
    path = os.path.join(WORKDIR, f"{uuid.uuid4().hex}.png")
    img.save(path)
    return path

def load_image(path: str) -> dict:
    img = Image.open(path).convert("RGB")
    return {"image_path": _save(img), "summary": f"loaded {os.path.basename(path)}"}

def resize(image_path: str, width: int, height: int) -> dict:
    img = Image.open(image_path).convert("RGB").resize((width, height))
    return {"image_path": _save(img), "summary": f"resized to {width}x{height}"}

def canny_edges(image_path: str, low: int=50, high: int=150) -> dict:
    img = cv2.imread(image_path)[:, :, ::-1]
    edges = cv2.Canny(img, low, high)
    edges_rgb = np.stack([edges]*3, axis=-1)
    out = Image.fromarray(edges_rgb)
    return {"image_path": _save(out), "summary": f"canny edges low={low} high={high}"}

def draw_bbox(image_path: str, x1:int, y1:int, x2:int, y2:int, color:str="red") -> dict:
    img = Image.open(image_path).convert("RGB")
    drw = ImageDraw.Draw(img)
    drw.rectangle([x1, y1, x2, y2], outline=color, width=4)
    return {"image_path": _save(img), "summary": f"drew box {x1,y1,x2,y2} {color}"}

TOOLS = {
    "load_image": ToolSpec(
        name="load_image",
        description="Load an image from a local path.",
        args_schema={"type":"object","properties":{"path":{"type":"string"}}, "required":["path"]},
    ),
    "resize": ToolSpec(
        name="resize",
        description="Resize the current image to width x height.",
        args_schema={"type":"object","properties":{"image_path":{"type":"string"},"width":{"type":"integer"},"height":{"type":"integer"}}, "required":["image_path","width","height"]},
    ),
    "canny_edges": ToolSpec(
        name="canny_edges",
        description="Run Canny edge detector on image.",
        args_schema={"type":"object","properties":{"image_path":{"type":"string"},"low":{"type":"integer"},"high":{"type":"integer"}}, "required":["image_path"]},
    ),
    "draw_bbox": ToolSpec(
        name="draw_bbox",
        description="Draw a rectangle on image.",
        args_schema={"type":"object","properties":{"image_path":{"type":"string"},"x1":{"type":"integer"},"y1":{"type":"integer"},"x2":{"type":"integer"},"y2":{"type":"integer"},"color":{"type":"string"}}, "required":["image_path","x1","y1","x2","y2"]},
    ),
}

def run_tool(name: str, **kwargs) -> dict:
    if name == "load_image":   return load_image(**kwargs)
    if name == "resize":       return resize(**kwargs)
    if name == "canny_edges":  return canny_edges(**kwargs)
    if name == "draw_bbox":    return draw_bbox(**kwargs)
    raise ValueError(f"Unknown tool {name}")
