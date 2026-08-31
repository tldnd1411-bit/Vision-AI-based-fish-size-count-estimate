# datasets/Flatfish.py
import os, random, torch, numpy as np, cv2, glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as standard_transforms
import warnings
warnings.filterwarnings('ignore')


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    # 이미지 로드
    img_cv = cv2.imread(img_path)
    H, W = img_cv.shape[:2]                     # 정규화 좌표 복원용
    img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    # YOLO 포맷 라벨 파싱
    points = []
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                # parts[0]=class, [1:5]=bbox(xc,yc,w,h), [5:7]=keypoint(x,y)
                kx, ky = float(parts[5]), float(parts[6])
                abs_x = kx * W
                abs_y = ky * H
                points.append([abs_y, abs_x])    # PET은 (y, x) 순서 요구!
    points = np.array(points, dtype=np.float64) if len(points) > 0 else np.zeros((0, 2))
    return img, points


class Oliver_Flounder(Dataset):
    def __init__(self, data_root, transform=None, train=False, flip=False, num_patches=8):
        self.root_path = data_root
        prefix = "train_data" if train else "test_data"
        self.prefix = prefix
        self.img_list = os.listdir(f"{data_root}/{prefix}/images")
        self.gt_list = {}
        for img_name in self.img_list:
            img_path = f"{data_root}/{prefix}/images/{img_name}"
            base = os.path.splitext(img_name)[0]
            gt_path = f"{data_root}/{prefix}/labels/{base}.txt"   # YOLO 라벨 폴더명 = labels
            self.gt_list[img_path] = gt_path
        self.img_list = sorted(list(self.gt_list.keys()))
        self.nSamples = len(self.img_list)
        self.transform = transform
        self.train = train
        self.flip = flip
        self.patch_size = 256
        # __getitem__ 자체는 PET 원본과 완전히 동일하게 이미지 1장당 랜덤 크롭 1개(단일 dict
        # target)만 반환한다 -> util/misc.py의 collate_fn, engine.py 등 다른 파일은 손댈 필요가
        # 전혀 없다.
        # 대신 학습 시에는 __len__을 nSamples * num_patches로 늘려서, 같은 원본 이미지를
        # 여러 인덱스(가상 인덱스)에 매핑해 둔다. DataLoader가 이 가상 인덱스들을 서로 다른
        # 시점에 호출할 때마다 __getitem__이 매번 새로 랜덤 크롭을 수행하므로, 결과적으로
        # 한 epoch 동안 이미지 한 장에서 num_patches개의 서로 다른 패치가 학습에 쓰이게 된다.
        self.num_patches = num_patches if train else 1

    def compute_density(self, points):
        points_tensor = torch.from_numpy(points.copy())
        dist = torch.cdist(points_tensor, points_tensor, p=2)
        if points_tensor.shape[0] > 1:
            density = dist.sort(dim=1)[0][:, 1].mean().reshape(-1)
        else:
            density = torch.tensor(999.0).reshape(-1)
        return density

    def __len__(self):
        return self.nSamples * self.num_patches

    def __getitem__(self, index):
        # 가상 인덱스 -> 실제 이미지 인덱스로 매핑 (num_patches개의 가상 인덱스가 같은 이미지를 가리킴)
        real_index = index % self.nSamples

        img_path = self.img_list[real_index]
        gt_path = self.gt_list[img_path]
        img, points = load_data((img_path, gt_path), self.train)
        points = points.astype(float)

        if self.transform is not None:
            img = self.transform(img)
        img = torch.Tensor(img)

        if self.train:
            scale_range = [0.8, 1.2]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            if scale * min_size > self.patch_size:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
            points *= scale

        if self.train:
            img, points = random_crop(img, points, patch_size=self.patch_size)

        if random.random() > 0.5 and self.train and self.flip:
            img = torch.flip(img, dims=[2])
            points[:, 1] = self.patch_size - points[:, 1]

        target = {}
        target['points'] = torch.Tensor(points)
        target['labels'] = torch.ones([points.shape[0]]).long()
        if self.train:
            target['density'] = self.compute_density(points)
        if not self.train:
            target['image_path'] = img_path

        return img, target


def random_crop(img, points, patch_size=256):
    patch_h = patch_w = patch_size
    start_h = random.randint(0, img.size(1) - patch_h) if img.size(1) > patch_h else 0
    start_w = random.randint(0, img.size(2) - patch_w) if img.size(2) > patch_w else 0
    end_h, end_w = start_h + patch_h, start_w + patch_w
    if points.shape[0] > 0:
        idx = (points[:, 0] >= start_h) & (points[:, 0] <= end_h) & \
              (points[:, 1] >= start_w) & (points[:, 1] <= end_w)
        result_points = points[idx]
        result_points[:, 0] -= start_h
        result_points[:, 1] -= start_w
    else:
        result_points = points
    result_img = img[:, start_h:end_h, start_w:end_w]
    imgH, imgW = result_img.shape[-2:]
    fH, fW = patch_h / imgH, patch_w / imgW
    result_img = torch.nn.functional.interpolate(result_img.unsqueeze(0), (patch_h, patch_w)).squeeze(0)
    if result_points.shape[0] > 0:
        result_points[:, 0] *= fH
        result_points[:, 1] *= fW
    return result_img, result_points


def build(image_set, args):
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data_root = args.data_path
    num_patches = getattr(args, 'num_patches', 8)
    if image_set == 'train':
        return Oliver_Flounder(data_root, train=True, transform=transform, flip=True, num_patches=num_patches)
    elif image_set == 'val':
        return Oliver_Flounder(data_root, train=False, transform=transform)