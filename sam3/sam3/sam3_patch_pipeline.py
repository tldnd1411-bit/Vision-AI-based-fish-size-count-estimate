# %%
# ============================================================
# STEP 0. 환경 확인 (torch / cuda / bf16) + SAM3 모델 로드
# ============================================================
import os

import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("bf16 supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)

import matplotlib
matplotlib.use("Agg")  # headless(shell) 환경: 창을 띄우지 않고 파일로만 저장. pyplot import 전에 설정.
import matplotlib.pyplot as plt
import numpy as np
import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# use bfloat16 for the entire script
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../sam3/sam3
BPE_VOCAB_PATH = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "assets", "bpe_simple_vocab_16e6.txt.gz")
)

bpe_path = BPE_VOCAB_PATH
model = build_sam3_image_model(bpe_path=bpe_path)
print("model device:", next(model.parameters()).device)


# %%
# ============================================================
# STEP 1. 지정 프레임 기준 1초 median 이미지 계산 + 샤프닝 (파일 저장만 수행)
# ============================================================
"""
지정한 프레임(frame_idx)을 기준으로 1초 동안의 프레임들을 읽어와
픽셀 단위 중앙값(median) 이미지를 계산한 뒤, 샤프닝(sharpening)을 적용해
결과를 파일로 저장하는 스크립트. (headless 환경이라 화면 표시는 하지 않음)

사용 라이브러리: opencv-python, numpy
설치: pip install opencv-python numpy
"""
import cv2


def compute_median_frame(video_path: str, frame_idx: int, duration_sec: float = 1.0,
                          mode: str = "forward"):
    """
    video_path 에서 frame_idx를 기준으로 duration_sec(초) 동안의 프레임들을 읽어
    픽셀별 중앙값 이미지를 계산한다.

    Parameters
    ----------
    video_path : str
        영상 파일 경로
    frame_idx : int
        기준이 되는 프레임 인덱스
    duration_sec : float
        중앙값을 계산할 구간의 길이(초). 기본 1초.
    mode : str
        "forward"  -> frame_idx 부터 duration_sec 만큼 뒤로 (frame_idx ~ frame_idx + N-1)
        "center"   -> frame_idx 를 중심으로 앞뒤 duration_sec/2 씩
        "backward" -> frame_idx 에서 duration_sec 만큼 앞으로 (frame_idx - N + 1 ~ frame_idx)

    Returns
    -------
    median_img : np.ndarray (H, W, 3), dtype=uint8, RGB 순서
    used_frame_indices : list[int]
    fps : float
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not fps or fps <= 0:
        cap.release()
        raise ValueError("영상의 FPS 정보를 읽을 수 없습니다.")

    n_frames = max(1, round(fps * duration_sec))

    if mode == "forward":
        start = frame_idx
        end = frame_idx + n_frames - 1
    elif mode == "backward":
        start = frame_idx - n_frames + 1
        end = frame_idx
    elif mode == "center":
        half = n_frames // 2
        start = frame_idx - half
        end = start + n_frames - 1
    else:
        cap.release()
        raise ValueError(f"알 수 없는 mode: {mode}")

    # 영상 범위를 벗어나지 않도록 클리핑
    start = max(0, start)
    end = min(total_frames - 1, end)

    frames = []
    used_indices = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for idx in range(start, end + 1):
        ret, frame = cap.read()
        if not ret:
            break
        # BGR(OpenCV 기본) -> RGB 변환 후 저장
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        used_indices.append(idx)
    cap.release()

    if not frames:
        raise RuntimeError("지정한 구간에서 프레임을 하나도 읽지 못했습니다.")

    stack = np.stack(frames, axis=0)  # (N, H, W, 3)
    median_img = np.median(stack, axis=0).astype(np.uint8)
    return median_img, used_indices, fps


def sharpen_image(img: np.ndarray, method: str = "unsharp_mask",
                   amount: float = 1.5, radius: float = 2.0, threshold: int = 0) -> np.ndarray:
    """
    이미지에 샤프닝을 적용한다.

    Parameters
    ----------
    img : np.ndarray
        입력 이미지 (H, W, 3), dtype=uint8
    method : str
        "unsharp_mask" -> 언샵 마스킹 (가우시안 블러 기반, 자연스러운 결과, 기본값)
        "kernel"       -> 고정 커널 컨볼루션 방식 (더 강하고 거친 샤프닝)
    amount : float
        샤프닝 강도. unsharp_mask 방식에서 사용. 클수록 강하게 적용됨 (예: 1.0~3.0).
    radius : float
        unsharp_mask 방식의 가우시안 블러 반경(sigma). 클수록 넓은 영역의 대비를 강조.
    threshold : int
        unsharp_mask 방식에서, 원본과 블러 이미지의 차이가 이 값보다 작은
        (즉 변화가 미미한, 노이즈일 가능성이 있는) 픽셀은 샤프닝하지 않음.

    Returns
    -------
    sharpened : np.ndarray (H, W, 3), dtype=uint8
    """
    if method == "kernel":
        kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(img, -1, kernel)
        return sharpened
    elif method == "unsharp_mask":
        img_f = img.astype(np.float32)
        blurred = cv2.GaussianBlur(img_f, ksize=(0, 0), sigmaX=radius)
        low_contrast_mask = np.abs(img_f - blurred) < threshold
        sharpened = img_f + amount * (img_f - blurred)
        sharpened = np.clip(sharpened, 0, 255)
        if threshold > 0:
            sharpened = np.where(low_contrast_mask, img_f, sharpened)
        return sharpened.astype(np.uint8)
    else:
        raise ValueError(f"알 수 없는 method: {method}")


def save_image(img: np.ndarray, save_path: str) -> str:
    """
    RGB 순서의 uint8 이미지를 파일로 저장한다. (내부적으로 BGR로 변환하여 cv2.imwrite 사용)

    Parameters
    ----------
    img : np.ndarray
        (H, W, 3), dtype=uint8, RGB 순서
    save_path : str
        저장할 파일 경로 (예: "./sharpened_frame_8000.png")

    Returns
    -------
    save_path : str
        저장에 성공한 경로 (그대로 반환)
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(save_path, bgr)
    if not ok:
        raise IOError(f"이미지 저장에 실패했습니다: {save_path}")
    print(f"이미지 저장 완료: {save_path}")
    return save_path


def compute_and_save_median_frame(video_path: str, frame_idx: int, duration_sec: float = 1.0,
                                   mode: str = "forward", sharpen_method: str = "unsharp_mask",
                                   sharpen_amount: float = 1.5, sharpen_radius: float = 2.0,
                                   save_dir: str = ".", save_median: bool = True,
                                   save_sharpened: bool = True):
    median_img, used_indices, fps = compute_median_frame(
        video_path, frame_idx, duration_sec=duration_sec, mode=mode
    )
    sharpened_img = sharpen_image(
        median_img, method=sharpen_method, amount=sharpen_amount, radius=sharpen_radius
    )

    print(f"FPS: {fps:.3f}")
    print(f"사용된 프레임 수: {len(used_indices)}")
    print(f"사용된 프레임 범위: {used_indices[0]} ~ {used_indices[-1]}")
    print(f"샤프닝 방식: {sharpen_method} (amount={sharpen_amount}, radius={sharpen_radius})")

    if save_median:
        median_path = os.path.join(save_dir, f"median_frame_{frame_idx}.png")
        save_image(median_img, median_path)
    if save_sharpened:
        sharpened_path = os.path.join(save_dir, f"sharpened_frame_{frame_idx}.png")
        save_image(sharpened_img, sharpened_path)

    return median_img, sharpened_img


# TODO: Linux 환경의 실제 경로로 바꿔주세요.
video_path = "/home/user/다운로드/넙치 카메라-20260511-154209-1778481729039-7.mp4"
frame_idx = 500

# mode="forward": frame_idx 부터 1초 동안(기본값)
# mode="center" 로 바꾸면 frame_idx를 중심으로 앞뒤 0.5초씩 사용
#
# sharpen_method="unsharp_mask" (기본, 자연스러움) 또는 "kernel" (더 강한 샤프닝)
# sharpen_amount, sharpen_radius 로 강도를 조절
#
# save_dir 에 median/sharpened 이미지가 각각
# median_frame_{frame_idx}.png / sharpened_frame_{frame_idx}.png 로 저장됩니다.
median_img, sharpened_img = compute_and_save_median_frame(
    video_path, frame_idx, duration_sec=1.0, mode="forward",
    sharpen_method="unsharp_mask", sharpen_amount=1.5, sharpen_radius=2.0,
    save_dir=".", save_median=True, save_sharpened=True,
)


# %%
# ============================================================
# STEP 2. 이미지를 겹치는 패치로 잘라 SAM3 text-prompt 추론 후 합치기
# ============================================================
"""
SAM3 text-prompt 기반 multi-prompt segmentation을, 이미지를 겹치는 패치로 잘라
"패치 단위"로 추론한 뒤 전체 이미지 좌표로 합치는 버전.

왜 패치로 나누면 도움이 될 수 있는가
------------------------------------
Sam3Processor는 내부적으로 입력 이미지를 고정 해상도(기본 1008x1008)로
리사이즈해서 vision backbone에 넣습니다. 우리 프레임은 크롭 후에도
~2840x2160 정도라서, 통째로 넣으면 가로 기준 약 2.8배 다운스케일됩니다.

그러면:
  - 넙치 한 마리(원본에서 40~60px)가 1008 해상도에서는 15~20px 수준으로
    작아져서 grounding head가 "물고기" 개별 형태를 인식하기 더 어려워지고,
  - "floor"/"yellow floor"는 원래도 물고기에 대부분 가려진 채 화면 전체에
    자잘하게 흩어진 비정형(amorphous) 영역인데, 다운스케일까지 겹치면
    질감/경계가 뭉개져서 텍스트 프롬프트가 매칭할 만한 "바닥처럼 보이는
    덩어리"가 거의 안 남습니다.
  - 게다가 SAM3 grounding head는 기본적으로 (COCO/LVIS류 학습 특성상)
    경계가 뚜렷한 개별 객체(thing)를 잘 찾도록 학습되어 있어서,
    "floor"처럼 화면 전체에 걸친 배경성 영역(stuff)은 원래도 텍스트
    prompt 매칭이 약한 편입니다. 패치로 나누면 완전히 해결되진 않아도,
    각 패치 안에서는 floor가 화면에서 차지하는 "비율"이 커지고 다운스케일
    손실도 줄어서, 최소한 일부 패치(특히 물고기가 적고 바닥이 넓게 노출된
    우측 영역 등)에서는 인식이 훨씬 잘 될 가능성이 높습니다.

주의: 이 방법으로도 "floor"가 전혀 안 잡히면, SAM3(모델)가 애초에
'광범위한 배경 영역' 자체를 텍스트로 잘 못 찾는 것일 수 있습니다. 그럴 땐
이전에 드린 OpenCV(black-hat 기반) classical CV 스크립트가 대안입니다 —
실제 이미지로 검증했을 때 그 방식은 물고기 vs 바닥을 꽤 잘 구분했습니다.
"""
import matplotlib.patches as patches

# =========================
# 입력 이미지 설정
# =========================
IMAGE_PATH = "./sharpened_frame_500.png"

if IMAGE_PATH is not None:
    img0 = Image.open(IMAGE_PATH).convert("RGB")
    print(f"이미지 파일에서 로드: {IMAGE_PATH}")
else:
    if "sharpened_img" not in dir():
        raise NameError(
            "sharpened_img가 정의되어 있지 않습니다. STEP 1을 먼저 실행해서 "
            "샤프닝된 이미지를 만들거나 IMAGE_PATH를 지정하세요."
        )
    arr = sharpened_img
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    img0 = Image.fromarray(arr).convert("RGB")
    print("메모리 상의 sharpened_img 변수를 사용합니다.")

print("image size (원본):", img0.size)

# =========================
# 좌우 / 하단 크롭
# =========================
# CROP_LEFT_RIGHT: 좌우 각각 이 픽셀만큼 잘라냄
# CROP_BOTTOM: 하단에서 이 픽셀만큼 잘라냄 (0이면 하단 크롭 안 함)
CROP_LEFT_RIGHT = 0
CROP_BOTTOM = 0

_orig_w, _orig_h = img0.size
img0 = img0.crop((
    CROP_LEFT_RIGHT,
    0,
    _orig_w - CROP_LEFT_RIGHT,
    _orig_h - CROP_BOTTOM,
))
print("image size (크롭 후):", img0.size)

if "model" not in dir():
    raise NameError(
        "model이 정의되어 있지 않습니다. STEP 0(build_sam3_image_model)을 "
        "먼저 실행한 뒤 이 셀을 실행하세요."
    )

processor = Sam3Processor(model, confidence_threshold=0.05)

prompt_configs = [
    #    {"prompt": "pipe", "threshold": 0.1},
    #    {"prompt": "yellow wall", "threshold": 0.5},
    #    {"prompt": "yellow floor", "threshold": 0.2},
    #    {"prompt": "water", "threshold": 0.9},
    {"prompt": "grey rock", "threshold": 0.01},
    #    {"prompt": "grey gravel", "threshold": 0.1},
]

# =========================
# 패치 분할 설정
# =========================
# Sam3Processor 기본 처리 해상도가 1008이므로, 패치도 그 근처로 맞추면
# 패치 안에서 리사이즈로 인한 정보 손실이 최소화됩니다.
PATCH_SIZE = 1008
PATCH_OVERLAP = 200  # 패치 경계에서 잘리는 물체(floor 조각 등)를 줄이기 위한 여유


def make_patch_boxes(img_w, img_h, patch_size=PATCH_SIZE, overlap=PATCH_OVERLAP):
    """
    (x1, y1, x2, y2) 픽셀 좌표의 패치 목록을 만든다. 이미지 가장자리까지
    빠짐없이 덮도록 마지막 패치는 오른쪽/아래쪽 끝에 맞춰 당겨진다.
    """
    stride = max(1, patch_size - overlap)

    def _starts(total):
        if total <= patch_size:
            return [0]
        starts = list(range(0, total - patch_size + 1, stride))
        if starts[-1] != total - patch_size:
            starts.append(total - patch_size)
        return starts

    xs = _starts(img_w)
    ys = _starts(img_h)
    boxes = []
    for y in ys:
        for x in xs:
            x2 = min(x + patch_size, img_w)
            y2 = min(y + patch_size, img_h)
            boxes.append((x, y, x2, y2))
    return boxes


def run_prompt_on_patch(patch_img, prompt, threshold):
    """patch_img(PIL) 위에서 단일 text prompt 추론을 수행하고 결과를 반환."""
    state = processor.set_image(patch_img)
    processor.reset_all_prompts(state)
    state = processor.set_confidence_threshold(threshold, state)
    state = processor.set_text_prompt(state=state, prompt=prompt)
    return state["boxes"], state["scores"], state["masks"]


# =========================
# 패치 단위 추론 + 전체 이미지 좌표로 합치기
# =========================
W, H = img0.size
patch_boxes = make_patch_boxes(W, H)
print(f"패치 개수: {len(patch_boxes)} (patch_size={PATCH_SIZE}, overlap={PATCH_OVERLAP})")

all_boxes = []
all_scores = []
all_masks = []
all_labels = []

for cfg in prompt_configs:
    prompt = cfg["prompt"]
    thr = cfg["threshold"]

    # 이 prompt에 대해 전체 이미지 크기의 마스크를 patch들의 합집합(OR)으로 누적
    combined_mask_full = np.zeros((H, W), dtype=bool)
    combined_boxes_full = []
    combined_scores = []
    n_found_total = 0

    for (x1, y1, x2, y2) in patch_boxes:
        patch_img = img0.crop((x1, y1, x2, y2))
        pw, ph = patch_img.size

        boxes, scores, masks = run_prompt_on_patch(patch_img, prompt, thr)
        n = len(boxes)
        if n == 0:
            continue
        n_found_total += n

        boxes_np = boxes.detach().float().cpu().numpy()  # patch-local xyxy(px)
        scores_np = scores.detach().float().cpu().numpy()
        masks_np = masks.detach().float().cpu().numpy()  # (n,1,h,w) or (n,h,w)
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]

        # 박스: patch 원점만큼 offset
        for b, s in zip(boxes_np, scores_np):
            bx1, by1, bx2, by2 = b
            combined_boxes_full.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1])
            combined_scores.append(s)

        # 마스크: patch 크기에 맞춰 리사이즈 후, 전체 캔버스의 해당 위치에 OR로 합성
        for m in masks_np:
            mm = m > 0.5
            if mm.shape != (ph, pw):
                mm_img = Image.fromarray((mm.astype(np.uint8) * 255)).resize(
                    (pw, ph), resample=Image.NEAREST
                )
                mm = np.array(mm_img) > 127
            combined_mask_full[y1:y2, x1:x2] |= mm

    if n_found_total > 0:
        all_masks.append(torch.from_numpy(combined_mask_full.astype(np.float32))[None])  # (1,H,W)
        all_boxes.append(torch.tensor(combined_boxes_full, dtype=torch.float32) if combined_boxes_full
                          else torch.zeros((0, 4)))
        all_scores.append(torch.tensor(combined_scores, dtype=torch.float32) if combined_scores
                           else torch.zeros((0,)))
        all_labels.append(prompt)  # 패치들을 합쳤으므로 prompt당 마스크 1장으로 취급

    print(f'prompt="{prompt}" -> 패치 전체에서 {n_found_total}개 raw 검출 '
          f'(합쳐진 마스크 픽셀수={combined_mask_full.sum()})')


# =========================
# 결과 시각화 (bbox + segmentation) -> 파일로만 저장
# =========================
def visualize_multi_prompt_results(
        image, all_boxes, all_scores, all_masks, all_labels,
        box_format="xyxy", mask_alpha=0.45, save_path=None,
):
    img_np = np.array(image.convert("RGB"))
    H, W = img_np.shape[:2]

    if len(all_masks) == 0:
        print("탐지된 객체가 없습니다.")
        fig, ax = plt.subplots(1, 1, figsize=(W / 100, H / 100), dpi=100)
        ax.imshow(img_np)
        ax.axis("off")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
            print(f"결과 이미지 저장: {save_path}")
        plt.close(fig)
        return

    unique_labels = sorted(set(all_labels))
    cmap = plt.get_cmap("tab10")
    label_to_color = {lbl: cmap(i % 10)[:3] for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(1, 1, figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(img_np)

    overlay = np.zeros((H, W, 4), dtype=np.float32)
    for label, mask_full in zip(all_labels, all_masks):
        color = label_to_color[label]
        mask = mask_full[0].numpy() > 0.5
        overlay[mask] = (*color, mask_alpha)
    ax.imshow(overlay)

    for label, boxes in zip(all_labels, all_boxes):
        color = label_to_color[label]
        for (x1, y1, x2, y2) in boxes.numpy():
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.2, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)

    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        print(f"결과 이미지 저장: {save_path}")
    plt.close(fig)


visualize_multi_prompt_results(
    img0, all_boxes, all_scores, all_masks, all_labels,
    box_format="xyxy",
    save_path=f"./frame_{frame_idx}_patched_multi_prompt.jpg",
)


# %%
# ============================================================
# STEP 3. 탐지된 영역(segmentation)만 남기고 나머지는 배경 제거 -> 파일로만 저장
# ============================================================
"""
[바로 이전 STEP: 패치 단위 SAM3 multi-prompt 추론] 다음 단계.

이전 단계에서 탐지된 segmentation(all_masks/all_labels, 예: "grey gravel")의
합집합을 "유지할 영역"으로 보고, 그 외 나머지(탐지되지 않은 부분 = 물고기,
사육기, 벽 등)는 전부 배경으로 간주해 투명 처리(제거)합니다.

즉 이전에 만든 스크립트들과 반대 방향입니다:
  - sam3_remove_background.py 등: "탐지된 부분(배경)"을 지우고 나머지를 유지
  - 이 스크립트: "탐지된 부분"만 유지하고 나머지를 지움

전제: img0, all_masks, all_labels, frame_idx 가 이전 STEP에서 이미 만들어져
있어야 합니다.
"""

# =========================
# 사전 조건 체크
# =========================
for _var in ("img0", "all_masks", "all_labels", "frame_idx"):
    if _var not in dir():
        raise NameError(
            f"'{_var}'가 정의되어 있지 않습니다. "
            "바로 이전 STEP(패치 단위 SAM3 multi-prompt 추론)을 먼저 실행하세요."
        )

img_np = np.array(img0.convert("RGB"))
H, W = img_np.shape[:2]

# =========================
# 탐지된 모든 라벨의 마스크를 합집합(OR) -> "유지할 영역"
# =========================
keep_mask = np.zeros((H, W), dtype=bool)
if len(all_masks) == 0:
    print("탐지된 객체가 없습니다 — 전체가 배경으로 처리(전부 투명)됩니다.")
else:
    for label, mask_full in zip(all_labels, all_masks):
        m = mask_full[0].numpy() > 0.5  # (H, W)
        keep_mask |= m
        print(f'"{label}" 마스크 픽셀수: {int(m.sum())}')

remove_mask = ~keep_mask
print(f"유지되는 픽셀: {int(keep_mask.sum())} / {H * W} ({keep_mask.mean() * 100:.1f}%)")
print(f"배경으로 제거되는 픽셀: {int(remove_mask.sum())} / {H * W} ({remove_mask.mean() * 100:.1f}%)")

# =========================
# RGBA 결과 생성 (탐지 안 된 부분 = 투명) 후 파일로 저장
# =========================
rgba = np.dstack([img_np, np.full((H, W), 255, dtype=np.uint8)])
rgba[remove_mask, 3] = 0

save_path = f"./inference/kept_only_frame_{frame_idx}.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
Image.fromarray(rgba, mode="RGBA").save(save_path)
print(f"결과 저장: {save_path}")