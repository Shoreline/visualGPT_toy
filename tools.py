# tools.py
from PIL import Image, ImageDraw
import numpy as np, cv2, uuid, os
from pydantic import BaseModel, Field
from ultralytics import YOLO

WORKDIR = "artifacts"; os.makedirs(WORKDIR, exist_ok=True)

_yolo_models = {}  # 简单的内存缓存，避免每次都重新加载

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

def detect_objects(image_path: str, model_name: str = "yolov8n.pt",
                   conf: float = 0.25, max_det: int = 50) -> dict:
    """
    目标检测：返回画好框的图片，并把检测结果以 JSON 形式返回给 LLM 复用
    """
    # 加载或复用模型
    model = _yolo_models.get(model_name)
    if model is None:
        model = YOLO(model_name)  # 首次会自动下载权重
        _yolo_models[model_name] = model

    results = model.predict(image_path, conf=conf, max_det=max_det, verbose=False)
    res = results[0]

    # 在图上画框
    # (Ultralytics 已经提供可视化结果)
    plotted = res.plot()  # ndarray (H, W, 3), BGR
    img = Image.fromarray(plotted[:, :, ::-1])  # 转回 RGB
    out_path = _save(img)

    # 构建结构化检测结果（给 LLM 用）
    dets = []
    names = res.names  # 类别 id -> 名称
    if res.boxes is not None:
        for b in res.boxes:
            xyxy = b.xyxy[0].tolist()  # [x1,y1,x2,y2]
            cls_id = int(b.cls[0].item())
            score = float(b.conf[0].item()) if b.conf is not None else None
            dets.append({
                "bbox": [round(v, 2) for v in xyxy],
                "cls_id": cls_id,
                "cls_name": names.get(cls_id, str(cls_id)),
                "score": round(score, 3) if score is not None else None
            })

    summary = f"detected {len(dets)} objects @conf>={conf}"
    return {"image_path": out_path, "summary": summary, "detections": dets}

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
    "detect_objects": ToolSpec(
        name="detect_objects",
        description="Detect objects in image with YOLO model.",
        args_schema={
            "type":"object",
            "properties":{
                "image_path":{"type":"string"},
                "model_name":{"type":"string","default":"yolov8n.pt"},
                "conf":{"type":"number","default":0.25},
                "max_det":{"type":"integer","default":50}
            },
            "required":["image_path"]
        }        
    ),
}

def run_tool(name: str, **kwargs) -> dict:
    if name == "load_image":   return load_image(**kwargs)
    if name == "resize":       return resize(**kwargs)
    if name == "canny_edges":  return canny_edges(**kwargs)
    if name == "draw_bbox":    return draw_bbox(**kwargs)
    if name == "detect_objects": return detect_objects(**kwargs)
    raise ValueError(f"Unknown tool {name}")
