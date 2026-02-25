# src/obg/browser/vision.py
from __future__ import annotations

import base64
import io
import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from obg.llm.client import OllamaClient, LLMResponse


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class BBox:
    """Pixel-space bbox: (x1,y1) top-left, (x2,y2) bottom-right."""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str = ""
    score: Optional[float] = None

    def clamp(self, w: int, h: int) -> "BBox":
        x1 = max(0, min(self.x1, w - 1))
        y1 = max(0, min(self.y1, h - 1))
        x2 = max(0, min(self.x2, w - 1))
        y2 = max(0, min(self.y2, h - 1))
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        return BBox(x1, y1, x2, y2, self.label, self.score)

    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


@dataclass
class ScreenshotPrep:
    """Holds original + preprocessed images and resize mapping."""
    img_bgr: np.ndarray
    proc_bgr: np.ndarray
    scale_x: float
    scale_y: float


# -----------------------------
# Preprocessing
# -----------------------------

def decode_png_bytes(png_bytes: bytes) -> np.ndarray:
    """Decode PNG/JPEG bytes -> BGR image."""
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed; invalid image bytes?")
    return img


def preprocess_screenshot(
    img_bgr: np.ndarray,
    *,
    max_side: int = 1280,
    sharpen: bool = True,
    contrast: bool = True,
) -> ScreenshotPrep:
    """
    Resize (keeping aspect) to reduce token/image bandwidth to the LLM,
    and optionally apply light sharpening/contrast to help OCR-ish perception.
    Returns scale factors to map bbox coords back to original pixels.
    """
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    proc = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if contrast:
        # CLAHE on L channel in LAB
        lab = cv2.cvtColor(proc, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        proc = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

    if sharpen:
        # unsharp mask
        blur = cv2.GaussianBlur(proc, (0, 0), sigmaX=1.0, sigmaY=1.0)
        proc = cv2.addWeighted(proc, 1.5, blur, -0.5, 0)

    return ScreenshotPrep(
        img_bgr=img_bgr,
        proc_bgr=proc,
        scale_x=(w / proc.shape[1]),
        scale_y=(h / proc.shape[0]),
    )


def encode_png_base64(img_bgr: np.ndarray) -> str:
    """Encode BGR -> PNG bytes -> base64 string (for Ollama images)."""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# -----------------------------
# LLM prompt for boxes
# -----------------------------

VISION_BOX_SYSTEM = """\
You are a vision assistant for a web UI screenshot.
Return ONLY valid JSON.

Task:
- Given a GOAL and a screenshot, find the UI element(s) relevant to the goal.
- Output pixel-space bounding boxes in the image you received.

Bounding box format:
{
  "boxes": [
    {"label": "...", "x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>, "score": <float 0..1 optional>}
  ]
}

Rules:
- Coordinates are pixels in the provided image (top-left origin).
- x1,y1 is top-left; x2,y2 is bottom-right.
- Use tight boxes around the clickable target.
- If unsure, output an empty list: {"boxes": []}
"""

def make_vision_box_user(goal: str, extra_hints: Optional[str] = None) -> str:
    hints = f"\nHINTS:\n{extra_hints}\n" if extra_hints else ""
    return f"""\
GOAL: {goal}
{hints}
Return JSON now.
"""


def parse_boxes_json(text: str) -> List[BBox]:
    """
    Strictly parse {"boxes":[...]} from LLM output.
    Tolerates ```json fences and leading/trailing whitespace.
    """
    t = text.strip()
    t = re.sub(r"^\s*```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t).strip()

    try:
        obj = json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if not m:
            raise ValueError(f"LLM returned non-JSON: {text[:2000]}")
        obj = json.loads(m.group(0))

    boxes = obj.get("boxes", [])
    out: List[BBox] = []
    if not isinstance(boxes, list):
        return out
    for b in boxes:
        if not isinstance(b, dict):
            continue
        try:
            out.append(
                BBox(
                    x1=int(b["x1"]), y1=int(b["y1"]),
                    x2=int(b["x2"]), y2=int(b["y2"]),
                    label=str(b.get("label", "")),
                    score=(float(b["score"]) if "score" in b and b["score"] is not None else None),
                )
            )
        except Exception:
            continue
    return out


# -----------------------------
# Asking Ollama (Qwen3-VL)
# -----------------------------

def llm_find_boxes_on_screenshot(
    llm: OllamaClient,
    *,
    goal: str,
    prep: ScreenshotPrep,
    extra_hints: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> Tuple[List[BBox], LLMResponse]:
    """
    Sends preprocessed screenshot to Qwen3-VL via Ollama /api/chat
    and asks for bounding boxes in that (preprocessed) image space.
    """
    img_b64 = encode_png_base64(prep.proc_bgr)
    resp = llm.chat(
        messages=[
            {"role": "system", "content": VISION_BOX_SYSTEM},
            {"role": "user", "content": make_vision_box_user(goal, extra_hints), "images": [img_b64]},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    boxes = parse_boxes_json(resp.text)
    # Clamp in proc-image space
    ph, pw = prep.proc_bgr.shape[:2]
    boxes = [b.clamp(pw, ph) for b in boxes]
    return boxes, resp


def map_boxes_to_original(prep: ScreenshotPrep, boxes_proc: List[BBox]) -> List[BBox]:
    """
    Map boxes from preprocessed image space back to original image space.
    """
    oh, ow = prep.img_bgr.shape[:2]
    out: List[BBox] = []
    for b in boxes_proc:
        x1 = int(round(b.x1 * prep.scale_x))
        x2 = int(round(b.x2 * prep.scale_x))
        y1 = int(round(b.y1 * prep.scale_y))
        y2 = int(round(b.y2 * prep.scale_y))
        out.append(BBox(x1, y1, x2, y2, b.label, b.score).clamp(ow, oh))
    return out


# -----------------------------
# Drawing boxes
# -----------------------------

def draw_boxes(
    img_bgr: np.ndarray,
    boxes: List[BBox],
    *,
    thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """
    Draw boxes + labels on a copy of the image. (No specific colors requirement;
    OpenCV default is BGR; pick a couple of distinct ones.)
    """
    out = img_bgr.copy()
    palette = [(0, 255, 0), (255, 0, 0), (0, 128, 255), (255, 0, 255)]
    for i, b in enumerate(boxes):
        color = palette[i % len(palette)]
        cv2.rectangle(out, (b.x1, b.y1), (b.x2, b.y2), color, thickness)
        label = b.label or f"box{i}"
        if b.score is not None:
            label = f"{label} ({b.score:.2f})"
        # label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        x = b.x1
        y = max(0, b.y1 - th - 6)
        cv2.rectangle(out, (x, y), (x + tw + 6, y + th + 6), color, -1)
        cv2.putText(out, label, (x + 3, y + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# -----------------------------
# Verification vs BrowserGym
# -----------------------------

def iou(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


def browsergym_bbox_from_extra_props(
    extra_element_properties: List[Dict[str, Any]],
    *,
    target_browsergym_id: str,
) -> Optional[BBox]:
    """
    BrowserGym MiniWoB often provides per-element bbox info via obs["extra_element_properties"].

    Expected element dict has:
      - "browsergym_id" or "bid"
      - bbox fields vary by version; support common patterns:
          * {"bbox": {"x":..,"y":..,"width":..,"height":..}}
          * {"bounding_box": {"x":..,"y":..,"w":..,"h":..}}
          * {"left":..,"top":..,"right":..,"bottom":..}
    """
    for e in extra_element_properties or []:
        bid = str(e.get("browsergym_id") or e.get("bid") or e.get("id") or "")
        if bid != str(target_browsergym_id):
            continue

        if "bbox" in e and isinstance(e["bbox"], dict):
            bb = e["bbox"]
            x = float(bb.get("x", 0)); y = float(bb.get("y", 0))
            w = float(bb.get("width", bb.get("w", 0))); h = float(bb.get("height", bb.get("h", 0)))
            return BBox(int(x), int(y), int(x + w), int(y + h), label=f"bid:{bid}")

        if "bounding_box" in e and isinstance(e["bounding_box"], dict):
            bb = e["bounding_box"]
            x = float(bb.get("x", 0)); y = float(bb.get("y", 0))
            w = float(bb.get("width", bb.get("w", 0))); h = float(bb.get("height", bb.get("h", 0)))
            return BBox(int(x), int(y), int(x + w), int(y + h), label=f"bid:{bid}")

        # left/top/right/bottom
        if all(k in e for k in ("left", "top", "right", "bottom")):
            return BBox(int(e["left"]), int(e["top"]), int(e["right"]), int(e["bottom"]), label=f"bid:{bid}")

    return None


def verify_llm_boxes_against_browsergym(
    llm_boxes: List[BBox],
    browsergym_target_bbox: BBox,
    *,
    iou_threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Compares a set of LLM-predicted boxes against a BrowserGym bbox.
    Returns best match and pass/fail.
    """
    best = None
    best_iou = 0.0
    for b in llm_boxes:
        s = iou(b, browsergym_target_bbox)
        if s > best_iou:
            best_iou = s
            best = b
    return {
        "pass": bool(best is not None and best_iou >= iou_threshold),
        "best_iou": best_iou,
        "best_box": None if best is None else best.__dict__,
        "target_box": browsergym_target_bbox.__dict__,
        "threshold": iou_threshold,
    }


# -----------------------------
# End-to-end helper
# -----------------------------

def locate_and_verify(
    llm: OllamaClient,
    *,
    goal: str,
    screenshot_png_bytes: bytes,
    extra_element_properties: List[Dict[str, Any]],
    target_browsergym_id: str,
    max_side: int = 1280,
    iou_threshold: float = 0.3,
    extra_hints: Optional[str] = None,
) -> Dict[str, Any]:
    """
    1) preprocess screenshot
    2) ask LLM for bbox(es) in preprocessed space
    3) map boxes back to original
    4) pull target bbox from BrowserGym (extra_element_properties)
    5) compute IoU + return diagnostics + annotated image bytes (PNG)

    Returns:
      {
        "verification": {...},
        "llm_boxes": [...],
        "target_bbox": {...},
        "annotated_png": <bytes>,
        "latency_ms": int,
        "raw_llm": {...}
      }
    """
    img = decode_png_bytes(screenshot_png_bytes)
    prep = preprocess_screenshot(img, max_side=max_side)

    boxes_proc, llm_resp = llm_find_boxes_on_screenshot(
        llm,
        goal=goal,
        prep=prep,
        extra_hints=extra_hints,
        temperature=0.0,
        max_tokens=256,
    )
    boxes_orig = map_boxes_to_original(prep, boxes_proc)

    tgt = browsergym_bbox_from_extra_props(extra_element_properties, target_browsergym_id=target_browsergym_id)
    if tgt is None:
        raise ValueError(f"Could not find bbox for browsergym_id={target_browsergym_id} in extra_element_properties")

    ver = verify_llm_boxes_against_browsergym(boxes_orig, tgt, iou_threshold=iou_threshold)

    annotated = draw_boxes(img, [tgt] + boxes_orig)  # target first, then preds
    ok, buf = cv2.imencode(".png", annotated)
    if not ok:
        raise ValueError("Failed to encode annotated PNG")

    return {
        "verification": ver,
        "llm_boxes": [b.__dict__ for b in boxes_orig],
        "target_bbox": tgt.__dict__,
        "annotated_png": buf.tobytes(),
        "latency_ms": llm_resp.latency_ms,
        "raw_llm": llm_resp.raw,
        "llm_text": llm_resp.text,
    }

    # Example usage (in your BrowserGym loop)
#
# - screenshot_png_bytes: get from Playwright page.screenshot(type="png") OR from env wrapper if you already record frames
# - extra_element_properties: obs.get("extra_element_properties", [])
# - target_browsergym_id: the BID you believe is correct (e.g., from AXTree role=button "no")
#
# from obg.llm.client import OllamaClient
# from obg.browser.vision import locate_and_verify
#
# llm = OllamaClient(host="http://127.0.0.1:11434", model="qwen3-vl:latest")
# result = locate_and_verify(
#     llm,
#     goal=goal,
#     screenshot_png_bytes=png_bytes,
#     extra_element_properties=obs.get("extra_element_properties", []),
#     target_browsergym_id="13",
#     iou_threshold=0.3,
# )
# print(result["verification"])
# open("/tmp/annotated.png","wb").write(result["annotated_png"])

#	•	Coordinate convention: this assumes BrowserGym bbox is already in screenshot pixel coordinates. If your screenshot is scaled (deviceScaleFactor, viewport resizing), keep them consistent (best: screenshot at the same viewport scale as BrowserGym’s bbox).
#	•	If BrowserGym returns bbox in CSS pixels and your screenshot is device pixels, you’ll see IoU drift; fix by multiplying BrowserGym bbox by deviceScaleFactor.
#	•	extra_element_properties bbox keys vary; if your dict uses different keys, tell me one sample element and I’ll extend browsergym_bbox_from_extra_props() to match it.