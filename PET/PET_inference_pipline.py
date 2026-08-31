import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as standard_transforms
import util.misc as utils
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)
    # model parameters
    # - backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'fourier'),
                        help="Type of positional embedding to use on top of the image features")
    # - transformer
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    # loss parameters
    # - matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="SmoothL1 point coefficient in the matching cost")
    # - loss coefficients
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)       # classification loss coefficient
    parser.add_argument('--point_loss_coef', default=5.0, type=float)    # regression loss coefficient
    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")   # cross-entropy weights
    # dataset parameters
    parser.add_argument('--dataset_file', default="SHA")
    parser.add_argument('--data_path', default="./data/ShanghaiTech/PartA", type=str)
    # misc parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--vis_dir', default="")
    parser.add_argument('--num_workers', default=2, type=int)
    # --- 예측 point를 YOLO 스타일 txt로 저장하기 위한 옵션 ---
    parser.add_argument('--pred_label_dir', default="",
                        help="예측 point를 YOLO 스타일 txt로 저장할 폴더. "
                             "비워두면(default) 저장하지 않음.")
    parser.add_argument('--score_thresh', default=0.5, type=float,
                        help="이 값보다 confidence score가 높은 point만 '유효한 예측'으로 채택")
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def get_label_path(img_path):
    """
    이미지 경로에서 GT 라벨(txt) 경로를 유추합니다.
    'images' 폴더 컴포넌트를 'labels'로 바꾸고, 확장자를 .txt로 바꿉니다.
    예:
      .../test_data/images/F2_..._0004.jpg
      -> .../test_data/labels/F2_..._0004.txt
    폴더 규칙이 다르면 이 함수만 고치면 됩니다.
    """
    parts = img_path.split('/')
    parts = ['labels' if p == 'images' else p for p in parts]
    label_path = '/'.join(parts)
    label_path = os.path.splitext(label_path)[0] + '.txt'
    return label_path


def load_gt_points(gt_path, img_w, img_h):
    """
    라벨 txt 파일에서 GT 포인트(점) 좌표를 읽어 픽셀 좌표로 반환합니다.
    한 줄의 형식: 'class cx cy w h dot_x dot_y' (총 7개 값, YOLO 스타일)
      - cx, cy, w, h   : 정규화된(0~1) bbox 중심/크기 (여기서는 사용하지 않음)
      - dot_x, dot_y   : 정규화된(0~1) 실제 점(dot) 좌표 -> 우리가 찍을 GT 점
    예:
      0 0.639583 0.013333 0.022500 0.026667 0.643333 0.013333
      -> dot_x=0.643333, dot_y=0.013333 (이미지 가로/세로 기준 정규화)
    혹시 다른 파일이 'x y' 2개 값짜리 픽셀 좌표 형식으로 섞여 있어도
    처리할 수 있도록 fallback을 남겨뒀습니다.
    img_w, img_h: 정규화 좌표를 픽셀로 되돌릴 기준 이미지 크기.
    """
    points = []
    if not os.path.exists(gt_path):
        print('경고: GT 라벨 파일을 찾을 수 없습니다:', gt_path)
        return points
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.replace(',', ' ').split()
            try:
                if len(tokens) >= 7:
                    # class cx cy w h dot_x dot_y (정규화 좌표)
                    dot_x_norm = float(tokens[5])
                    dot_y_norm = float(tokens[6])
                    x = dot_x_norm * img_w
                    y = dot_y_norm * img_h
                    points.append([x, y])
                elif len(tokens) == 2:
                    # 'x y' 픽셀 좌표 (fallback)
                    x, y = float(tokens[0]), float(tokens[1])
                    points.append([x, y])
                else:
                    continue
            except ValueError:
                continue
    return points


def get_pred_label_path(img_path, pred_label_dir):
    """
    예측 point를 저장할 txt 경로를 만듭니다.
    GT 라벨(get_label_path)과 겹쳐서 실수로 덮어쓰지 않도록,
    별도의 pred_label_dir 아래에 이미지와 같은 basename으로 저장합니다.
    예:
      img_path = .../test_data/images/F2_..._0004.jpg
      pred_label_dir = ./inference_results/pred_labels
      -> ./inference_results/pred_labels/F2_..._0004.txt
    """
    name = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(pred_label_dir, name + '.txt')


def save_pred_points_yolo(pred_points_xy, ref_w, ref_h, save_path,
                           class_id=0, box_size_norm=0.01):
    """
    예측된 point 좌표(픽셀, (x, y) 순서 리스트)를 YOLO 스타일 txt로 저장합니다.
    load_gt_points()가 파싱하는 형식과 완전히 동일하게
    'class cx cy w h dot_x dot_y' (7개 값)로 저장하므로,
    이 함수로 저장한 txt를 load_gt_points()로 다시 읽어서 검증/비교에 쓸 수 있습니다.
    PET 모델은 bbox를 예측하지 않고 point만 예측하므로, cx/cy/w/h 4개 값은
    실제 bbox 회귀값이 아니라 dot_x/dot_y를 중심으로 한 더미(placeholder)
    작은 박스입니다.
    좌표 정규화 기준: ref_w/ref_h — 이 함수를 호출하는 쪽에서 넘겨주는 '기준 해상도'입니다.
    (원본이 3840x2160으로 들어와 크롭+리사이즈 파이프라인을 탄 경우, 이 기준은 원본이
    아니라 크롭된 '작업 해상도'(예: 3340x2060)입니다. 그래야 이 txt를 그 크롭된 이미지
    위에서 바로 읽었을 때 별도의 크롭 보정 없이 좌표가 정확히 맞습니다.)
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    lines = []
    for (x_px, y_px) in pred_points_xy:
        dot_x = x_px / ref_w
        dot_y = y_px / ref_h
        # 혹시 패딩 등으로 살짝 범위를 벗어나는 경우를 대비해 0~1로 clip
        dot_x = min(max(dot_x, 0.0), 1.0)
        dot_y = min(max(dot_y, 0.0), 1.0)
        cx, cy = dot_x, dot_y
        w = h = box_size_norm
        lines.append(
            f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {dot_x:.6f} {dot_y:.6f}"
        )
    with open(save_path, 'w') as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")
    print(f"예측 point {len(lines)}개를 YOLO 스타일로 저장 (기준 해상도 {ref_w}x{ref_h}): {save_path}")
    return save_path


def add_title_bar(img, text, bar_height=32, bg_color=(40, 40, 40), text_color=(255, 255, 255)):
    """
    이미지 위쪽에 제목(GT/Pred, 개수)을 표시하는 띠를 붙여서 반환합니다.
    """
    h, w = img.shape[:2]
    bar = np.full((bar_height, w, 3), bg_color, dtype=np.uint8)
    cv2.putText(bar, text, (10, bar_height - 9), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, text_color, 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def visualization(base_img_bgr, pred_xy, gt_xy, vis_dir, img_path, split_map=None):
    """
    '작업 해상도'(크롭+리사이즈 파이프라인을 탔다면 크롭된 3340x2060, 아니면 원본) 이미지인
    base_img_bgr(BGR, np.uint8) 위에 GT/Pred를 각각 그린 뒤, 좌우로 나란히 붙여서 저장합니다.

    pred_xy, gt_xy: 둘 다 (x_px, y_px) 좌표의 리스트/이터러블. base_img_bgr과 같은
    좌표계(=작업 해상도 기준)여야 합니다.
    """
    gt_vis = base_img_bgr.copy()
    pred_vis = base_img_bgr.copy()

    size = max(3, int(round(min(base_img_bgr.shape[:2]) / 300)))
    for x, y in gt_xy:
        gt_vis = cv2.circle(gt_vis, (int(round(x)), int(round(y))), size, (0, 0, 255), -1)
    for x, y in pred_xy:
        pred_vis = cv2.circle(pred_vis, (int(round(x)), int(round(y))), size, (0, 255, 0), -1)

    if split_map is not None:
        imgH, imgW = pred_vis.shape[:2]
        split_map_color = (split_map * 255).astype(np.uint8)
        split_map_color = cv2.applyColorMap(split_map_color, cv2.COLORMAP_JET)
        split_map_color = cv2.resize(split_map_color, (imgW, imgH), interpolation=cv2.INTER_NEAREST)
        pred_vis = (split_map_color * 0.9 + pred_vis).clip(0, 255).astype(np.uint8)

    if vis_dir is not None:
        pred_cnt = len(pred_xy)
        gt_cnt = len(gt_xy)
        gt_vis = add_title_bar(gt_vis, 'GT: {}'.format(gt_cnt))
        pred_vis = add_title_bar(pred_vis, 'Pred: {}'.format(pred_cnt))
        divider = np.full((gt_vis.shape[0], 4, 3), 255, dtype=np.uint8)
        compare_vis = np.hstack([gt_vis, divider, pred_vis])
        name = img_path.split('/')[-1].split('.')[0]
        img_save_path = os.path.join(
            vis_dir, '{}_compare_gt{}_pred{}.jpg'.format(name, gt_cnt, pred_cnt)
        )
        cv2.imwrite(img_save_path, compare_vis)
        print('image save to ', img_save_path)
        print('pred count: {}, gt count: {}, |error|: {}'.format(
            pred_cnt, gt_cnt, abs(pred_cnt - gt_cnt)))


# ---------------------------------------------------------------------------
# 크롭 + 멀티 해상도 추론 설정
#   원본이 정확히 3840x2160으로 들어오면:
#     1) 좌우 CROP_LEFT_RIGHT px씩, 하단 CROP_BOTTOM px 크롭 -> '작업 해상도'
#        (기본 3340x2060)를 만들고,
#     2) 그 작업 이미지를 다시 (INFER_WIDTH x INFER_HEIGHT)로 리사이즈해서 추론합니다.
#   이후 예측 point 저장/시각화는 원본(3840x2160)이 아니라 '작업 해상도'(3340x2060)
#   기준으로 합니다 — SAM2 등 후속 파이프라인이 쓰는 크롭된 이미지와 좌표계가
#   바로 일치하도록 하기 위해서입니다.
#   원본이 3840x2160이 아니면 크롭/리사이즈 없이 원본 그대로 추론합니다(기존 동작 유지).
# ---------------------------------------------------------------------------
SRC_W, SRC_H = 3840, 2160
CROP_LEFT_RIGHT = 250
CROP_BOTTOM = 100
INFER_WIDTH = 1280
INFER_HEIGHT = 720


def crop_to_working_size(img_bgr, crop_left_right=CROP_LEFT_RIGHT, crop_bottom=CROP_BOTTOM):
    """원본 이미지를 좌우/하단 크롭해서 '작업 해상도' 이미지를 만든다."""
    h, w = img_bgr.shape[:2]
    left = crop_left_right if crop_left_right > 0 else 0
    right = w - crop_left_right if crop_left_right > 0 else w
    bottom = h - crop_bottom if crop_bottom > 0 else h
    if left >= right or bottom <= 0:
        print(f"[경고] CROP 설정(좌우={crop_left_right}, 하단={crop_bottom})이 이미지 크기"
              f"({w}x{h})보다 커서 크롭을 건너뜁니다.")
        return img_bgr
    cropped = img_bgr[0:bottom, left:right]
    return cropped


@torch.no_grad()
def evaluate_single_image(model, img_path, device, vis_dir=None,
                           pred_label_dir=None, score_thresh=0.5):
    model.eval()
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    # load image (원본 그대로 읽음)
    img_bgr_raw = cv2.imread(img_path)
    raw_h, raw_w = img_bgr_raw.shape[:2]

    if (raw_w, raw_h) == (SRC_W, SRC_H):
        # 3840x2160 원본 -> 크롭해서 작업 해상도(예: 3340x2060) 생성
        img_bgr = crop_to_working_size(img_bgr_raw, CROP_LEFT_RIGHT, CROP_BOTTOM)
        work_h, work_w = img_bgr.shape[:2]
        print(f"[크롭] {raw_w}x{raw_h} -> {work_w}x{work_h} "
              f"(좌우 {CROP_LEFT_RIGHT}px씩, 하단 {CROP_BOTTOM}px)")
        infer_w, infer_h = INFER_WIDTH, INFER_HEIGHT
        img_for_infer_bgr = cv2.resize(img_bgr, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
        print(f"[리사이즈] {work_w}x{work_h} -> {infer_w}x{infer_h} 로 리사이즈해서 추론합니다.")
    else:
        # 원본이 3840x2160이 아니면 크롭/리사이즈 없이 그대로 추론(기존 동작과 동일)
        img_bgr = img_bgr_raw
        work_h, work_w = raw_h, raw_w
        infer_w, infer_h = raw_w, raw_h
        img_for_infer_bgr = img_bgr

    # 이후 모든 좌표(저장/시각화/GT 로딩)의 기준 해상도 = '작업 해상도'
    orig_w, orig_h = work_w, work_h

    img = Image.fromarray(cv2.cvtColor(img_for_infer_bgr, cv2.COLOR_BGR2RGB))

    # transform image
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    img = torch.Tensor(img)
    samples = utils.nested_tensor_from_tensor_list([img])
    samples = samples.to(device)
    padded_h, padded_w = samples.tensors.shape[-2:]  # 모델에 실제로 들어가는(패딩 포함) 캔버스 크기

    # load GT points (라벨 txt의 정규화 좌표 -> '작업 해상도' 픽셀 좌표)
    gt_path = get_label_path(img_path)
    gt_points_xy = load_gt_points(gt_path, orig_w, orig_h)
    print('GT label path: ', gt_path)
    print('GT count: ', len(gt_points_xy))

    # inference
    outputs = model(samples, test=True)
    raw_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)
    outputs_scores = raw_scores[:, :, 1][0]
    outputs_points = outputs['pred_points'][0]
    print('prediction (raw query 수): ', len(outputs_scores))

    keep = outputs_scores > score_thresh
    kept_points = outputs_points[keep]
    print(f'prediction (score > {score_thresh} 필터링 후, 저장 대상): ', len(kept_points))

    # 리사이즈해서 추론한 만큼 다시 확대해서 '작업 해상도'(orig_w x orig_h) 좌표로 복원.
    # infer_w == orig_w, infer_h == orig_h 이면(크롭/리사이즈 안 한 경우) scale=1.0 이라
    # 기존 코드와 동일한 결과가 나옵니다.
    scale_x = orig_w / infer_w
    scale_y = orig_h / infer_h

    def to_work_xy(norm_point):
        # norm_point = (norm_y, norm_x): 모델 출력 그대로의 인덱싱 순서
        x_px = float(norm_point[1]) * padded_w * scale_x
        y_px = float(norm_point[0]) * padded_h * scale_y
        return x_px, y_px

    pred_points_all_xy = [to_work_xy(p) for p in outputs_points]  # 시각화용 (전체 query)
    pred_points_px = [to_work_xy(p) for p in kept_points]         # 저장용 (score 필터링 후)

    if pred_label_dir:
        pred_txt_path = get_pred_label_path(img_path, pred_label_dir)
        # 저장 기준 해상도 = '작업 해상도'(orig_w, orig_h). 3840x2160 입력이었다면 3340x2060.
        save_pred_points_yolo(pred_points_px, orig_w, orig_h, pred_txt_path)

    # visualize predictions vs GT ('작업 해상도' 이미지 위에 그림)
    if vis_dir:
        split_map = (outputs['split_map_raw'][0].detach().cpu().squeeze(0) > 0.5).float().numpy()
        visualization(img_bgr, pred_points_all_xy, gt_points_xy, vis_dir, img_path, split_map=split_map)


def main(args):
    # input image and model
    args.img_path = './sharpened_frame_500.png'
    args.resume = './outputs/Oliver_Flounder/pet_model/best_checkpoint.pth'
    args.vis_dir = './inference_results'
    args.pred_label_dir = './inference_results/pred_labels'
    # build model
    device = torch.device(args.device)
    model, criterion = build_model(args)
    model.to(device)
    # load pretrained model
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    # evaluation
    vis_dir = None if args.vis_dir == "" else args.vis_dir
    pred_label_dir = None if args.pred_label_dir == "" else args.pred_label_dir
    evaluate_single_image(model, args.img_path, device, vis_dir=vis_dir,
                           pred_label_dir=pred_label_dir, score_thresh=args.score_thresh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
