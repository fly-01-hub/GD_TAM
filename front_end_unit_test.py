import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import json

# ==========================================
# 0. 环境变量严格注入 (杜绝底层导包异常)
# ==========================================
PROJECT_ROOT = os.path.expanduser("~/project/GD_TAM")
sys.path.append(os.path.join(PROJECT_ROOT, "EfficientTAM"))
sys.path.append(os.path.join(PROJECT_ROOT, "GroundingDINO")) # 双保险，确保 DINO 的底层调用不出错

# ==========================================
# 1. 路径与参数配置 (Configuration)
# ==========================================
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
FRAME_DIR = os.path.join(PROJECT_ROOT, "data/test_frames")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/test_results")

# 强制保障输出目录存在 (补漏：防止 cv2.imwrite 静默失败)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# DINO 配置
DINO_CONFIG = os.path.join(PROJECT_ROOT, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
DINO_WEIGHTS = os.path.join(CHECKPOINT_DIR, "groundingdino_swint_ogc.pth")
TEXT_PROMPT = "person"  # 务必确保此提示词与你的测试视频内容匹配
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25    #当目标特征不够明显时可降低这两个值来放宽检测条件

# EfficientTAM 配置
SAM2_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "efficienttam_s.pt")
SAM2_CONFIG = "configs/efficienttam/efficienttam_s.yaml"

# 环境设置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. 模型加载与环境探针 (Model Initialization)
# ==========================================
print("[INFO] 正在加载 GroundingDINO...")
try:
    from groundingdino.util.inference import load_model as load_dino_model, predict as dino_predict
    import groundingdino.datasets.transforms as T
except ImportError as e:
    print(f"[FATAL] GroundingDINO 导入失败，请检查源码路径: {e}")
    sys.exit(1)

dino_model = load_dino_model(DINO_CONFIG, DINO_WEIGHTS)
dino_model = dino_model.to(DEVICE)

print("[INFO] 正在加载 EfficientTAM...")
try:
    from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor
except ImportError as e:
    print(f"[FATAL] EfficientTAM 导入失败，请检查接口名: {e}")
    sys.exit(1)

sam2_predictor = build_efficienttam_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)

# ==========================================
# 3. 首帧目标检测 (Detection on Frame 0)
# ==========================================
# 严格过滤，只读取图片文件，防止系统隐藏文件干扰
frame_names = sorted([f for f in os.listdir(FRAME_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

if not frame_names:
    print(f"[FATAL] 目录 {FRAME_DIR} 为空。FFmpeg 抽帧必定失败，请停止实验并排查数据源。")
    sys.exit(1)

first_frame_path = os.path.join(FRAME_DIR, frame_names[0])
first_frame_cv2 = cv2.imread(first_frame_path)

if first_frame_cv2 is None:
    print(f"[FATAL] 无法读取首帧图像，请检查文件权限或图像是否损坏: {first_frame_path}")
    sys.exit(1)

H, W, _ = first_frame_cv2.shape

transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
image_source = Image.open(first_frame_path).convert("RGB")
image_tensor, _ = transform(image_source, None)

print(f"[INFO] 正在首帧上执行全局搜索: '{TEXT_PROMPT}'")
boxes, logits, phrases = dino_predict(
    model=dino_model,
    image=image_tensor,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

if len(boxes) == 0:
    print(f"[ERROR] 视觉前端瘫痪：未能检测到 '{TEXT_PROMPT}'。请降低 BOX_THRESHOLD 或检查提示词。")
    sys.exit(1)

# 提取最高置信度标定框并转为绝对坐标系
best_box = boxes[0].numpy()
cx, cy, w, h = best_box * np.array([W, H, W, H])
x1, y1 = int(cx - w/2), int(cy - h/2)
x2, y2 = int(cx + w/2), int(cy + h/2)
dino_box_xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)

print(f"[INFO] 目标锁定: {phrases[0]} (置信度: {logits[0]:.2f})，初始状态: {dino_box_xyxy}")

# ==========================================
# 4. 时空传播与几何约束抽象 (Propagation & Fitting)
# ==========================================
inference_state = sam2_predictor.init_state(video_path=FRAME_DIR)
sam2_predictor.reset_state(inference_state)

sam2_predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=1,
    box=dino_box_xyxy
)

trajectory_data = {}

for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(inference_state):
    frame_path = os.path.join(FRAME_DIR, frame_names[out_frame_idx])
    img_display = cv2.imread(frame_path)
    
    # 提取有效掩码空间
    mask = (out_mask_logits[0][0].cpu().numpy() > 0.0).astype(np.uint8)
    
    # 掩码可视化叠加
    colored_mask = np.zeros_like(img_display)
    colored_mask[mask == 1] = [0, 255, 0]
    img_display = cv2.addWeighted(img_display, 1.0, colored_mask, 0.4, 0)

    # 最小二乘法椭圆拟合
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipse_params = None
    
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        # 数学约束：拟合二次曲线至少需要 5 个独立离散点
        if len(max_contour) >= 5:
            ellipse = cv2.fitEllipse(max_contour)
            cv2.ellipse(img_display, ellipse, (0, 0, 255), 2)
            
            ellipse_params = {
                "x": float(ellipse[0][0]), "y": float(ellipse[0][1]),
                "a": float(ellipse[1][0]), "b": float(ellipse[1][1]),
                "theta": float(ellipse[2])
            }
            # 绘制几何中心
            cv2.circle(img_display, (int(ellipse_params["x"]), int(ellipse_params["y"])), 3, (255, 0, 0), -1)

    trajectory_data[f"frame_{out_frame_idx:05d}"] = ellipse_params
    
    # 持久化输出
    out_path = os.path.join(OUTPUT_DIR, f"result_{out_frame_idx:05d}.jpg")
    cv2.imwrite(out_path, img_display)
    
    # 降低 I/O 打印频率以防终端拥堵，仅提供关键节点反馈
    if out_frame_idx % 50 == 0 or out_frame_idx == len(frame_names) - 1:
        print(f"[INFO] 时空传播进度: 帧 {out_frame_idx}/{len(frame_names)-1}")

with open(os.path.join(OUTPUT_DIR, "trajectory.json"), "w") as f:
    json.dump(trajectory_data, f, indent=4)

print(f"\n[SUCCESS] 视觉前端测试闭环。标定几何参数已序列化至: {OUTPUT_DIR}/trajectory.json")
