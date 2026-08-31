"""
fish_segmentation_pipeline.py

Jupyter 노트북 셀들을 하나의 스크립트로 합친 버전입니다.
SAM2로 물고기(도다리) point 라벨을 기반으로 segmentation을 수행하고,
- 검출/미검출 개체 수 추정
- 골격 기반 체장(cm) 추정
- 체장->체중 환산
까지 한 번에 실행합니다.

이 버전은 HDMI(디스플레이)가 없는 Jetson 등에서 실행하기 위해 matplotlib
시각화(figure 생성/표시) 코드를 전부 제거한 headless 버전입니다. 결과 이미지
(*_fish_removed.png)와 유효 픽셀 마스크(*_valid_mask.npy)는 그대로 파일로
저장되니, 확인이 필요하면 그 파일들을 다른 곳에서 열어보면 됩니다.

실행 전 확인할 것:
- 이 스크립트는 PET / sam3 / sam2 세 폴더가 형제(sibling)로 같은 상위 폴더 아래에 있고,
  이 파일 자신은 그중 sam2 폴더 안에 있다고 가정합니다.
      <BASE_DIR>/
      ├── PET/inference_results/pred_labels/sharpened_frame_500.txt
      ├── sam3/sam3/inference/kept_only_frame_500.png
      └── sam2/fish_segmentation_pipeline.py   <- 이 파일
  이 폴더 배치가 다르면 아래 IMAGE_PATH / LABEL_PATH / SAM2_REPO_PATH 계산 부분을 맞게 수정하세요.
- sam2, opencv-python(cv2), scikit-image, scipy, torch, pillow 패키지가 필요합니다.
"""

# ===== 셀 1: import =====
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Apple MPS에서 미지원 연산은 CPU로 폴백
import sys
import numpy as np
import torch
from PIL import Image
import heapq
from skimage.morphology import skeletonize
import cv2

# 이 파일(fish_segmentation_pipeline.py)이 있는 폴더 = <BASE_DIR>/sam2
# BASE_DIR = PET, sam3, sam2가 형제로 놓여있는 공통 상위 폴더.
# git clone 위치나 실행 시 현재 작업 디렉터리(cwd)와 무관하게 항상 올바른
# 경로를 찾도록, cwd가 아니라 이 파일 자신의 위치(__file__)를 기준으로 계산합니다.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_THIS_DIR)


# ===== 셀 2: device 선택 (원본 데모와 동일) =====
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")
if device.type == "cuda":
    # 노트북 전체에서 bfloat16 사용
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # Ampere 이상 GPU면 tf32 사용
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


# ===== 셀 3: 우리 프로젝트용 헬퍼 함수 (시각화 함수는 headless 실행을 위해 전부 제거) =====

def get_label_path(img_path):
    """이미지 경로 -> YOLO 스타일 point 라벨 txt 경로 ('images' -> 'labels', 확장자 .txt)."""
    parts = img_path.replace("\\", "/").split("/")
    parts = ["labels" if p == "images" else p for p in parts]
    label_path = "/".join(parts)
    label_path = os.path.splitext(label_path)[0] + ".txt"
    return label_path


def load_gt_points(gt_path, img_w, img_h):
    """
    YOLO 스타일 point 라벨 txt 로드.
    한 줄: 'class cx cy w h dot_x dot_y' (7개 값) -> dot_x,dot_y(정규화)를 픽셀로 반환.
    'x y' 2개 값짜리 픽셀 좌표 포맷도 fallback으로 지원.
    """
    points = []
    if not os.path.exists(gt_path):
        print("경고: 라벨 파일을 찾을 수 없습니다:", gt_path)
        return points
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.replace(",", " ").split()
            try:
                if len(tokens) >= 7:
                    dot_x_norm = float(tokens[5])
                    dot_y_norm = float(tokens[6])
                    x = dot_x_norm * img_w
                    y = dot_y_norm * img_h
                    points.append((x, y))
                elif len(tokens) == 2:
                    x, y = float(tokens[0]), float(tokens[1])
                    points.append((x, y))
                else:
                    continue
            except ValueError:
                continue
    return points


def load_image_apply_bg_removal(image_path, fill_color=(0, 0, 0)):
    """
    sam3_based_rmbg.py 등에서 배경을 투명(alpha=0)으로 지워 저장한 RGBA
    이미지를 읽어서, 제거된 픽셀을 fill_color로 채운 RGB 이미지로 돌려줍니다.
    (.convert("RGB")만 쓰면 alpha가 사라지며 지워진 배경이 다시 보이는
    문제가 있어서 직접 처리합니다.) alpha 채널이 없으면 원본을 그대로 사용.
    """
    img_pil = Image.open(image_path)
    has_alpha = (
        img_pil.mode in ("RGBA", "LA")
        or (img_pil.mode == "P" and "transparency" in img_pil.info)
    )
    if not has_alpha:
        print(f"'{image_path}'에 alpha 채널이 없어 배경 제거 정보를 찾을 수 없습니다. "
              f"원본 RGB를 그대로 사용합니다.")
        return img_pil.convert("RGB")
    arr = np.array(img_pil.convert("RGBA"))
    alpha = arr[..., 3]
    rgb = arr[..., :3].copy()
    removed = alpha == 0
    print(f"배경 제거된(alpha=0) 픽셀: {int(removed.sum())} / {removed.size} "
          f"({removed.mean() * 100:.1f}%) -> RGB를 {fill_color}로 채움")
    rgb[removed] = fill_color
    return Image.fromarray(rgb, mode="RGB")


def clip_mask_to_max_size(mask, center_x, center_y, img_w, img_h, max_size_px):
    """
    point 하나로 얻은 mask가 (center_x, center_y) 기준 max_size_px x max_size_px
    범위를 넘어서면, 그 범위 밖은 잘라냅니다.
    여러 객체(물고기)가 붙어 있어서 point 하나가 그 뭉치 전체를 한 객체로
    잡아버리는 문제를 막기 위한 상한선입니다.
    max_size_px가 None이면 아무 제한도 하지 않고 원본 mask를 그대로 반환합니다.
    """
    if max_size_px is None:
        return mask
    half = max_size_px / 2
    x0 = int(max(0, round(center_x - half)))
    x1 = int(min(img_w, round(center_x + half)))
    y0 = int(max(0, round(center_y - half)))
    y1 = int(min(img_h, round(center_y + half)))
    clipped = np.zeros_like(mask)
    clipped[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return clipped


def refine_box_from_mask(mask, img_w, img_h, pad_ratio=0.3, max_size_px=None):
    """mask의 실제 foreground를 감싸는 tight box를 구하고 pad + 상한을 적용."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    w, h = x1 - x0, y1 - y0
    x0 -= w * pad_ratio; x1 += w * pad_ratio
    y0 -= h * pad_ratio; y1 += h * pad_ratio
    if max_size_px is not None:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max_size_px / 2
        x0 = max(x0, cx - half); x1 = min(x1, cx + half)
        y0 = max(y0, cy - half); y1 = min(y1, cy + half)
    x0 = max(0.0, x0); y0 = max(0.0, y0)
    x1 = min(float(img_w), x1); y1 = min(float(img_h), y1)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def segment_point_sam2_adaptive(predictor, x, y, img_w, img_h,
                                 initial_box_size_px=30,
                                 max_object_size_px=40,
                                 refine_iters=2,
                                 pad_ratio=0.3):
    """point+box를 함께 넘겨 예측 -> mask 크기에 맞춰 box 재계산 -> 재추론, 반복."""
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    half = initial_box_size_px / 2
    box = np.array(
        [max(0, x - half), max(0, y - half), min(img_w, x + half), min(img_h, y + half)],
        dtype=np.float32,
    )
    mask, score = None, None
    for _ in range(refine_iters):
        masks, scores, _ = predictor.predict(
            point_coords=input_point, point_labels=input_label,
            box=box, multimask_output=False,
        )
        mask = masks[0] > 0.5
        score = float(scores[0])
        refined_box = refine_box_from_mask(mask, img_w, img_h, pad_ratio, max_object_size_px)
        if refined_box is None:
            break
        box = refined_box
    return mask, score, box


def smooth_mask(mask, kernel_size=5, keep_largest_only=True):
    """
    mask 경계가 울퉁불퉁하게 나올 때 부드럽게 다듬는 후처리입니다.
      1) morphological closing(작은 구멍 메움) + opening(작은 돌기 제거)
      2) Gaussian blur 후 재이진화로 경계를 한 번 더 부드럽게
      3) keep_largest_only=True면 mask 안에 떨어져 생긴 잔점들을 지우고
         가장 큰 덩어리 하나만 남깁니다.
    kernel_size가 클수록 더 매끈해지지만, 너무 크면 원래 형태(가느다란 부분 등)가
    뭉개질 수 있으니 3~9 사이에서 조절해보세요.
    """
    m = mask.astype(np.uint8) * 255
    k = max(1, kernel_size | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.GaussianBlur(m, (k, k), 0)
    m = (m > 127).astype(np.uint8) * 255
    if keep_largest_only:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = 1 + int(np.argmax(areas))
            m = np.where(labels == largest, 255, 0).astype(np.uint8)
    return m > 127


def compute_nearest_neighbor_dists(points):
    """각 point에서 가장 가까운 다른 point까지의 거리."""
    pts = np.array(points, dtype=np.float32)
    n = len(pts)
    if n <= 1:
        return np.full(n, np.inf, dtype=np.float32)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1)


def mask_contains_other_points(mask, points, self_idx):
    """mask 안에 자기 자신이 아닌 다른 point가 들어있으면 True (=합쳐짐 신호)."""
    H, W = mask.shape
    for j, (px, py) in enumerate(points):
        if j == self_idx:
            continue
        xi, yi = int(round(px)), int(round(py))
        if 0 <= yi < H and 0 <= xi < W and mask[yi, xi]:
            return True
    return False


def _extend_to_boundary(mask, tail, from_pt, step=0.5):
    """
    from_pt -> tail 방향의 직선을 tail에서부터 계속 연장하면서, mask를 벗어나기
    직전까지 이동한 거리와, 그 지점(연장된 끝점 좌표)을 함께 반환합니다.
    반환값: (extra_distance, (extended_y, extended_x))
    """
    ty, tx = tail
    fy, fx = from_pt
    dy, dx = ty - fy, tx - fx
    norm = (dy * dy + dx * dx) ** 0.5
    if norm == 0:
        return 0.0, (float(ty), float(tx))
    dy, dx = dy / norm, dx / norm
    H, W = mask.shape
    cy, cx = float(ty), float(tx)
    extra = 0.0
    while True:
        ny, nx = cy + dy * step, cx + dx * step
        iy, ix = int(round(ny)), int(round(nx))
        if not (0 <= iy < H and 0 <= ix < W) or not mask[iy, ix]:
            break
        cy, cx = ny, nx
        extra += step
    return extra, (cy, cx)


def _farthest_pair_on_skeleton(coord_set):
    """골격 픽셀 집합에서 서로 가장 먼 두 점(a, b)과 그 경로 길이를 구함."""
    if not coord_set:
        return None, None, 0.0

    def neighbors(p):
        y, x = p
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                q = (y + dy, x + dx)
                if q in coord_set:
                    yield q, (dy * dy + dx * dx) ** 0.5

    def farthest_point(src):
        dist = {src: 0.0}
        pq = [(0.0, src)]
        visited = set()
        best_node, best_dist = src, 0.0
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if d > best_dist:
                best_node, best_dist = u, d
            for v, w in neighbors(u):
                nd = d + w
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return best_node, best_dist

    start = next(iter(coord_set))
    a, _ = farthest_point(start)
    b, length_px = farthest_point(a)
    return a, b, length_px


def skeleton_length(mask):
    """mask의 실제 몸 길이(머리 끝~꼬리 끝)를 픽셀 단위로 추정."""
    skel = skeletonize(mask)
    ys, xs = np.where(skel)
    if len(xs) < 2:
        return 0.0
    coord_set = set(zip(ys.tolist(), xs.tolist()))
    a, b, length_px = _farthest_pair_on_skeleton(coord_set)
    if a is None:
        return 0.0
    extra_a, _ = _extend_to_boundary(mask, tail=a, from_pt=b)
    extra_b, _ = _extend_to_boundary(mask, tail=b, from_pt=a)
    length_px += extra_a + extra_b
    return length_px


def crop_by_margins(image, left_px=0, right_px=0, top_px=0, bottom_px=0):
    """
    keypoint(label)를 뽑을 때 이미지를 양옆/위아래로 잘라낸 뒤 정규화했다면,
    세그멘테이션용 이미지도 동일하게 잘라야 keypoint 위치가 맞습니다.
    예: 원본 3840x2160에서 좌우 250px씩, 아래 100px을 잘라서 keypoint를 만들었다면
        crop_by_margins(image, left_px=250, right_px=250, top_px=0, bottom_px=100)
        -> 결과 3340x2060 (기대한 크기와 일치)
    """
    h, w = image.shape[:2]
    x0, x1 = left_px, w - right_px
    y0, y1 = top_px, h - bottom_px
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"잘못된 crop 범위입니다: 원본={w}x{h}, "
            f"left={left_px}, right={right_px}, top={top_px}, bottom={bottom_px} "
            f"-> 결과 x=[{x0},{x1}), y=[{y0},{y1})"
        )
    return image[y0:y1, x0:x1]


def is_point_in_removed_area(image, x, y, threshold):
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < image.shape[1] and 0 <= yi < image.shape[0]):
        return True
    pixel = image[yi, xi]
    return int(pixel[0]) + int(pixel[1]) + int(pixel[2]) < threshold


def sl_to_weight(SL_cm, k=1.2, a_TL=0.01479, b=3.00):
    """
    체장(SL, cm) 단일 값 -> 체중(g) 변환
    k : TL/SL 비율 (TL = k * SL). 실측값 없으면 기본값 1.0 사용.
    a_TL, b : FishBase 도다리 전장(TL) 기준 계수 (a=0.01479, b=3.00)
    """
    a_SL = a_TL * (k ** b)
    return a_SL * (SL_cm ** b)


def main():
    # ===== 셀 4: 모델 빌드 + 우리 이미지 로드 =====
    # sam2 저장소 루트 = 이 스크립트가 들어있는 폴더 자체 (sam2/fish_segmentation_pipeline.py)
    SAM2_REPO_PATH = _THIS_DIR
    if SAM2_REPO_PATH not in sys.path:
        sys.path.insert(0, SAM2_REPO_PATH)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # PET, sam3와 sam2는 형제 폴더이므로 _BASE_DIR(공통 상위 폴더) 기준으로 계산
    IMAGE_PATH = os.path.join(_BASE_DIR, "sam3", "sam3", "inference", "kept_only_frame_500.png")
    LABEL_PATH = os.path.join(
        _BASE_DIR, "PET", "inference_results", "pred_labels", "sharpened_frame_500.txt"
    )

    # keypoint(sharpened_frame_500.txt)는 원본 3840x2160 프레임에서
    # 좌우 250px씩, 아래쪽 100px을 잘라낸(위쪽은 그대로 둔) 3340x2060 기준으로
    # 정규화되어 저장되어 있습니다. 세그멘테이션용 이미지도 동일하게 잘라야
    # keypoint 위치가 이미지와 정확히 맞습니다.
    CROP_LEFT_PX = 250
    CROP_RIGHT_PX = 250
    CROP_TOP_PX = 0
    CROP_BOTTOM_PX = 100

    sam2_checkpoint = os.path.join(SAM2_REPO_PATH, "checkpoints", "sam2.1_hiera_large.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    img0 = load_image_apply_bg_removal(IMAGE_PATH)
    image = np.array(img0)
    print("image size (원본, crop 전):", img0.size)
    image = crop_by_margins(
        image,
        left_px=CROP_LEFT_PX, right_px=CROP_RIGHT_PX,
        top_px=CROP_TOP_PX, bottom_px=CROP_BOTTOM_PX,
    )
    H, W = image.shape[:2]
    print("image size (crop 후, keypoint 기준과 일치해야 함):", (W, H))

    # ===== 셀 5: 라벨 로드 + 이미지 인코딩(한 번만) =====
    # 라벨 txt의 정규화 좌표를 원본(3840x2160)이 아니라, 지금 로드된 이 이미지(image)의
    # 실제 크기(W, H) 기준으로 그대로 복원합니다. (크롭 오프셋 계산/필터링 없음)
    label_path = LABEL_PATH if LABEL_PATH is not None else get_label_path(IMAGE_PATH)
    if not os.path.isabs(label_path):
        print(f"경고: LABEL_PATH가 절대경로가 아닙니다: {label_path}")
    points_px = load_gt_points(label_path, W, H)
    print(f"라벨 파일: {label_path}")
    print(f"로드된 point 개수: {len(points_px)}")
    predictor.set_image(image)  # 이미지 인코더는 여기서 딱 한 번만 실행됨

    # ===== 셀 8: 핵심 - point마다 adaptive box+point predict() 반복 호출 (headless: 시각화 없음) =====
    MAX_POINTS = 598
    INITIAL_BOX_SIZE_PX = 30
    MAX_OBJECT_SIZE_PX = 40
    REFINE_ITERS = 2
    SMOOTH_KERNEL_SIZE = 5
    SMOOTH_KEEP_LARGEST_ONLY = True
    BG_REMOVED_THRESHOLD = 10

    points_to_run = points_px if MAX_POINTS is None else points_px[:MAX_POINTS]
    if MAX_POINTS is not None and len(points_px) > MAX_POINTS:
        print(f"MAX_POINTS={MAX_POINTS}: 앞에서부터 {MAX_POINTS}개만 실행 "
              f"(나머지 {len(points_px) - MAX_POINTS}개 생략)")

    detected = 0
    skipped_removed = 0
    mask_pixel_counts = []
    all_masks = []
    skeleton_lengths_px = []

    for i, (x, y) in enumerate(points_to_run):
        if is_point_in_removed_area(image, x, y, threshold=BG_REMOVED_THRESHOLD):
            skipped_removed += 1
            print(f"point [{i}] = ({x:.1f}, {y:.1f}) -> RMBG로 제거된 배경 영역이라 무시")
            continue
        best_mask, best_score, final_box = segment_point_sam2_adaptive(
            predictor, x, y, W, H,
            initial_box_size_px=INITIAL_BOX_SIZE_PX,
            max_object_size_px=MAX_OBJECT_SIZE_PX,
            refine_iters=REFINE_ITERS,
        )
        best_mask = smooth_mask(
            best_mask,
            kernel_size=SMOOTH_KERNEL_SIZE,
            keep_largest_only=SMOOTH_KEEP_LARGEST_ONLY,
        )
        mask_px = int(best_mask.sum())
        mask_pixel_counts.append(mask_px)
        length_px = skeleton_length(best_mask)
        skeleton_lengths_px.append(length_px)
        all_masks.append(best_mask)
        print(f"point [{i}] = ({x:.1f}, {y:.1f}) -> score = {best_score:.3f}, "
              f"mask = {mask_px}px")
        detected += 1

    print(f"\n총 {len(points_to_run)}개 point 중 세그멘테이션 수행 {detected}개, "
          f"RMBG 배경이라 무시 {skipped_removed}개")

    # ===== 셀 10: 셀 8에서 segmentation된 물고기 영역을 검은색으로 덮고, 픽셀 데이터에서도 완전히 제외 =====
    RMBG_OUTPUT_PATH = os.path.splitext(IMAGE_PATH)[0] + "_fish_removed.png"
    VALID_MASK_OUTPUT_PATH = os.path.splitext(IMAGE_PATH)[0] + "_valid_mask.npy"

    combined_fish_mask = np.zeros((H, W), dtype=bool)
    for m in all_masks:
        combined_fish_mask |= m
    removed_px = int(combined_fish_mask.sum())
    print(f"segmentation된 전체 픽셀 수(제거 대상): {removed_px} / {H * W} "
          f"({combined_fish_mask.mean() * 100:.1f}%)")

    valid_mask = ~combined_fish_mask  # True = 실제 픽셀 데이터로 취급해야 하는 곳

    # 1) 저장용: segmentation된 영역을 배경과 동일한 완전한 검은색(0,0,0)으로 덮음
    FISH_REMOVED_FILL_COLOR = (0, 0, 0)
    result_arr = image.copy()
    result_arr[combined_fish_mask] = FISH_REMOVED_FILL_COLOR
    Image.fromarray(result_arr, mode="RGB").save(RMBG_OUTPUT_PATH)
    print(f"저장 완료(검은색 처리): {RMBG_OUTPUT_PATH}")

    # 2) 실제 픽셀값 계산에서 이 영역을 완전히 배제하기 위한 마스크
    #    (배경도 원래 검은색이라 색상값만으로는 "원래 배경"과 "방금 제거된 물고기"를
    #     구분할 수 없습니다 - 그래서 검은색 여부가 아니라 반드시 이 마스크로 걸러야 합니다.)
    np.save(VALID_MASK_OUTPUT_PATH, valid_mask)
    print(f"저장 완료(유효 픽셀 마스크): {VALID_MASK_OUTPUT_PATH}")

    def load_rmbg_result(img_path=RMBG_OUTPUT_PATH, mask_path=VALID_MASK_OUTPUT_PATH):
        """검은색으로 덮인 이미지를 다시 읽을 때, valid_mask로 걸러서 그 영역을 배열에서 아예 제외합니다."""
        rgb = np.array(Image.open(img_path).convert("RGB"))
        mask = np.load(mask_path)
        valid_pixels = rgb[mask]  # 제거된 영역은 여기 존재 자체가 없음
        return valid_pixels, rgb, mask

    example_valid_pixels, _, _ = load_rmbg_result()
    print(f"[마스크 기준] segmentation으로 제거된 영역만 제외한 유효 픽셀 수: "
          f"{example_valid_pixels.shape[0]} / {H * W} "
          f"({example_valid_pixels.shape[0] / (H * W) * 100:.1f}%)")

    # 색상값 기준으로 '검은색(0,0,0)이 아닌' 픽셀 수 - 원래부터 검었던 배경까지 포함해서 제외한 값이라
    # 위 마스크 기준 개수와는 다를 수 있습니다(원래 배경이 있었다면 이쪽이 더 적게 나옵니다).
    non_black_px = int(np.any(result_arr != 0, axis=-1).sum())
    black_px = H * W - non_black_px
    print(f"[색상 기준] 검은색(원래 배경 + segmentation 제거 영역 전부) 픽셀 수: {black_px} / {H * W}")
    print(f"[색상 기준] Unlabeled area: {non_black_px} / {H * W}px "
          f"({non_black_px / (H * W) * 100:.1f}%)")

    # ===== 셀 11: 골격 기반 길이(px)를 cm로 환산해 물고기 평균 길이 산출 =====
    CALIB_PX = 70.0
    CALIB_CM = 10.0
    PX_TO_CM = CALIB_CM / CALIB_PX

    # --- 이상치 제거 경계용 신뢰수준: 95%(z=1.96)는 경계가 느슨해서 이상치를 많이
    #     살려두므로, 더 타이트하게(기본 80%) 조입니다. 필요하면 0.68, 0.60 등으로
    #     더 조일 수 있습니다. (n이 충분히 크면 좁혀도 표본이 크게 줄지 않습니다)
    OUTLIER_CONFIDENCE = 0.80
    # --- 평균의 신뢰구간은 "이상치 제거"와는 다른 개념(추정치의 정밀도)이므로
    #     통계적 관례대로 95%를 그대로 유지합니다.
    MEAN_CI_CONFIDENCE = 0.95

    try:
        from scipy import stats
        Z_OUTLIER = stats.norm.ppf(0.5 + OUTLIER_CONFIDENCE / 2)
    except ImportError:
        _Z_TABLE = {0.60: 0.8416, 0.68: 1.0000, 0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600}
        Z_OUTLIER = _Z_TABLE.get(round(OUTLIER_CONFIDENCE, 2), 1.2816)

    lengths_px = np.array(skeleton_lengths_px, dtype=np.float64)
    lengths_px = lengths_px[lengths_px > 0]
    n = len(lengths_px)
    print(f"골격 길이를 구한 point 수: {n}")

    median_val = np.median(lengths_px)
    mad = np.median(np.abs(lengths_px - median_val))
    robust_std = 1.4826 * mad
    lower_bound = median_val - Z_OUTLIER * robust_std
    upper_bound = median_val + Z_OUTLIER * robust_std
    inlier = (lengths_px >= lower_bound) & (lengths_px <= upper_bound)
    filtered_px = lengths_px[inlier]
    print(f"[이상치 제거 경계 {OUTLIER_CONFIDENCE * 100:.0f}% (z={Z_OUTLIER:.3f})] "
          f"허용 범위=[{lower_bound:.1f}, {upper_bound:.1f}]px -> {n}개 중 {len(filtered_px)}개 유지 "
          f"({n - len(filtered_px)}개 제외)")

    filtered_cm = filtered_px * PX_TO_CM
    filtered_mean_cm = filtered_cm.mean()
    filtered_std_cm = filtered_cm.std(ddof=1)
    print(f"[이상치 제거 후] average fish length = {filtered_mean_cm:.2f}cm "
          f"(= {filtered_px.mean():.1f}px, n={len(filtered_cm)}, std={filtered_std_cm:.2f}cm)")

    sem_cm = filtered_std_cm / np.sqrt(len(filtered_cm))
    try:
        from scipy import stats
        crit = (stats.t.ppf(0.5 + MEAN_CI_CONFIDENCE / 2, df=len(filtered_cm) - 1)
                if len(filtered_cm) < 30
                else stats.norm.ppf(0.5 + MEAN_CI_CONFIDENCE / 2))
    except ImportError:
        crit = 1.9600 if MEAN_CI_CONFIDENCE >= 0.95 else 1.6449
    ci_half_width_cm = crit * sem_cm
    ci_low_cm, ci_high_cm = filtered_mean_cm - ci_half_width_cm, filtered_mean_cm + ci_half_width_cm
    print(f"평균 길이의 {MEAN_CI_CONFIDENCE * 100:.0f}% 신뢰구간 = [{ci_low_cm:.2f}, {ci_high_cm:.2f}]cm")

    # ===== 셀 12: 물고기 평균 segmentation 면적(px²) 산출 =====
    # 셀 11(골격 길이)과 동일한 방식 - 이상치 제거 경계와 평균의 신뢰구간을 분리해서 적용합니다.
    OUTLIER_CONFIDENCE = 0.80   # 이상치 제거 경계용 (95%는 느슨해서 타이트하게 80%로)
    MEAN_CI_CONFIDENCE = 0.95   # 평균의 신뢰구간은 통계 관례대로 95% 유지
    try:
        from scipy import stats
        Z_OUTLIER = stats.norm.ppf(0.5 + OUTLIER_CONFIDENCE / 2)
    except ImportError:
        _Z_TABLE = {0.60: 0.8416, 0.68: 1.0000, 0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600}
        Z_OUTLIER = _Z_TABLE.get(round(OUTLIER_CONFIDENCE, 2), 1.2816)

    areas_px = np.array(mask_pixel_counts, dtype=np.float64)
    areas_px = areas_px[areas_px > 0]
    n = len(areas_px)
    print(f"면적을 구한 point 수: {n}")

    median_val = np.median(areas_px)
    mad = np.median(np.abs(areas_px - median_val))
    robust_std = 1.4826 * mad
    lower_bound = median_val - Z_OUTLIER * robust_std
    upper_bound = median_val + Z_OUTLIER * robust_std
    inlier = (areas_px >= lower_bound) & (areas_px <= upper_bound)
    filtered_areas_px = areas_px[inlier]
    print(f"[이상치 제거 경계 {OUTLIER_CONFIDENCE * 100:.0f}% (z={Z_OUTLIER:.3f})] "
          f"허용 범위=[{lower_bound:.1f}, {upper_bound:.1f}]px² -> {n}개 중 {len(filtered_areas_px)}개 유지 "
          f"({n - len(filtered_areas_px)}개 제외)")

    filtered_mean_px = filtered_areas_px.mean()
    filtered_std_px = filtered_areas_px.std(ddof=1)
    print(f"[이상치 제거 후] fish average segmentation area = {filtered_mean_px:.1f}px² "
          f"(n={len(filtered_areas_px)}, std={filtered_std_px:.1f}px²)")

    sem_px = filtered_std_px / np.sqrt(len(filtered_areas_px))
    try:
        from scipy import stats
        crit = (stats.t.ppf(0.5 + MEAN_CI_CONFIDENCE / 2, df=len(filtered_areas_px) - 1)
                if len(filtered_areas_px) < 30
                else stats.norm.ppf(0.5 + MEAN_CI_CONFIDENCE / 2))
    except ImportError:
        crit = 1.9600 if MEAN_CI_CONFIDENCE >= 0.95 else 1.6449
    ci_half_width_px = crit * sem_px
    ci_low_px, ci_high_px = filtered_mean_px - ci_half_width_px, filtered_mean_px + ci_half_width_px
    print(f"평균 면적의 {MEAN_CI_CONFIDENCE * 100:.0f}% 신뢰구간 = [{ci_low_px:.1f}, {ci_high_px:.1f}]px²")

    # 남은 면적(라벨/검출되지 않은 영역) ÷ 마리당 평균 면적 = 추가로 있을 것으로 추정되는 마리 수
    Unlabeled = non_black_px / filtered_mean_px
    Total = detected + Unlabeled
    print(f"Deteted Count: {detected}")
    print(f"Area based Estimated Undetected Count: {Unlabeled:.2f}")
    print(f"Estimated Total Count: {Total:.1f}")

    # ===== 셀 13: 도다리 체장 체중 변환식 기반 중량 산출 =====
    avg_weight_g = sl_to_weight(filtered_mean_cm)
    print(f"Average SL: {filtered_mean_cm:.2f}cm, fish Weight: {avg_weight_g:.2f}g")


if __name__ == "__main__":
    main()