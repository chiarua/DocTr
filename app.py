"""DocTr 微服务：几何去弯曲 + 可选光照校正。
启动时一次性加载模型到 GPU，全局锁串行推理。
仅供本机 Go 后端调用，绑定 127.0.0.1:8501。
"""
import io
import os
import threading

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify, send_file

from inference import GeoTr_Seg, reload_model, reload_segmodel
from inference_ill import padCropImg, illCorrection, composePatch
from IllTr import IllTr

app = Flask(__name__)

# 全局推理锁：同时只允许一帧推理，避免 GPU 显存争用
inference_lock = threading.Lock()

# 模型路径（与原 inference.py 默认值一致，可用环境变量覆盖）
SEG_PATH = os.environ.get("DOCTR_SEG_PATH", "./model_pretrained/seg.pth")
GEOTR_PATH = os.environ.get("DOCTR_GEOTR_PATH", "./model_pretrained/geotr.pth")
ILLTR_PATH = os.environ.get("DOCTR_ILLTR_PATH", "./model_pretrained/illtr.pth")

# 启动时一次性加载模型
print(f"[doctr] loading models from SEG={SEG_PATH} GEO={GEOTR_PATH} ILL={ILLTR_PATH}", flush=True)

GeoTr_Seg_model = GeoTr_Seg().cuda()
reload_segmodel(GeoTr_Seg_model.msk, SEG_PATH)
reload_model(GeoTr_Seg_model.GeoTr, GEOTR_PATH)
GeoTr_Seg_model.eval()

IllTr_model = IllTr().cuda()
reload_model(IllTr_model, ILLTR_PATH)
IllTr_model.eval()

print("[doctr] models loaded, service ready", flush=True)


def rectify_geo(img_bgr_uint8: np.ndarray) -> np.ndarray:
    """几何去弯曲，输入输出均为 BGR uint8 ndarray。复用 inference.py 的 GeoTr_Seg。"""
    im_ori = img_bgr_uint8[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB
    h, w, _ = im_ori.shape
    im = cv2.resize(im_ori, (288, 288)).transpose(2, 0, 1)
    im = torch.from_numpy(im).float().unsqueeze(0)
    with torch.no_grad():
        bm = GeoTr_Seg_model(im.cuda()).cpu()
    bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))
    bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))
    bm0 = cv2.blur(bm0, (3, 3))
    bm1 = cv2.blur(bm1, (3, 3))
    lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)
    out = F.grid_sample(
        torch.from_numpy(im_ori).permute(2, 0, 1).unsqueeze(0).float(),
        lbl,
        align_corners=True,
    )
    return ((out[0] * 255).permute(1, 2, 0).numpy())[:, :, ::-1].astype(np.uint8)


def rectify_ill(img_bgr_uint8: np.ndarray) -> np.ndarray:
    """光照校正，复用 inference_ill.py 的 padCropImg/illCorrection/composePatch。"""
    totalPatch, padH, padW = padCropImg(img_bgr_uint8)
    totalResults = illCorrection(IllTr_model, totalPatch)
    return composePatch(totalResults, padH, padW, img_bgr_uint8)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/rectify")
def rectify():
    f = request.files.get("file")
    if f is None:
        return "missing file", 400
    ill_rec = request.form.get("ill_rec", "false").lower() in ("1", "true", "yes")

    try:
        pil = Image.open(f.stream).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return f"decode failed: {e}", 400

    with inference_lock:
        try:
            out = rectify_geo(img_bgr)
            if ill_rec:
                out = rectify_ill(out)
        except Exception as e:
            return f"inference failed: {e}", 500

    ok, buf = cv2.imencode(".png", out)
    if not ok:
        return "encode failed", 500
    return send_file(
        io.BytesIO(buf.tobytes()),
        mimetype="image/png",
        download_name="rectified.png",
    )


if __name__ == "__main__":
    # threaded=True 让 /health 不被 /rectify 阻塞（推理锁仍串行）
    app.run(host="127.0.0.1", port=8501, threaded=True, debug=False)
