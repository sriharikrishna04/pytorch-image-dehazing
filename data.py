import os
import random
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from utils import rgb_to_tensor


def augment(img_input, img_target):
    degree = random.choice([0, 90, 180, 270])
    if degree != 0:
        img_input = transforms.functional.rotate(img_input, degree)
        img_target = transforms.functional.rotate(img_target, degree)
    return img_input, img_target


def get_patch(img_input, img_target):
    w, h = img_input.size

    # Resize if smaller than 1024
    if w < 1024 or h < 1024:
        new_w = max(1024, w)
        new_h = max(1024, h)
        img_input = img_input.resize((new_w, new_h))
        img_target = img_target.resize((new_w, new_h))
        w, h = img_input.size

    # Decide patch size
    choice = random.choice([1, 2, 3])
    if choice == 1:
        p_x = p_y = 1024
    elif choice == 2:
        p_x, p_y = 1024, 2048
    else:
        p_x = p_y = 2048

    # ✅ Safety: ensure patch size is not larger than image
    p_x = min(p_x, w)
    p_y = min(p_y, h)

    # Crop safely
    x = random.randrange(0, max(1, w - p_x + 1))
    y = random.randrange(0, max(1, h - p_y + 1))
    img_input = img_input.crop((x, y, x + p_x, y + p_y))
    img_target = img_target.crop((x, y, x + p_x, y + p_y))

    # Resize large patches down to 1024x1024
    if p_x > 1024 or p_y > 1024:
        img_input = img_input.resize((1024, 1024))
        img_target = img_target.resize((1024, 1024))

    return img_input, img_target


def get_file_paths(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder)])


class MyDataset(data.Dataset):
    def __init__(self, data_dir, is_train=False):
        super(MyDataset, self).__init__()
        self.is_train = is_train

        hazy_dir = os.path.join(data_dir, "New Hazy dataset", "New Hazy dataset")
        gt_dir   = os.path.join(data_dir, "Ground Truth images", "Ground Truth images")

        hazy_files = sorted(os.listdir(hazy_dir))
        gt_files   = sorted(os.listdir(gt_dir))

        hazy_names = {os.path.splitext(f)[0]: f for f in hazy_files}
        gt_names   = {os.path.splitext(f)[0]: f for f in gt_files}

        # ✅ Keep only common files
        common = sorted(set(hazy_names.keys()) & set(gt_names.keys()))

        self.input_file_paths  = [os.path.join(hazy_dir, hazy_names[name]) for name in common]
        self.target_file_paths = [os.path.join(gt_dir, gt_names[name]) for name in common]

        print(f"✅ Using {len(self.input_file_paths)} paired images")

        self.n_samples = len(self.input_file_paths)

    def get_img_pair(self, idx):
        img_input = Image.open(self.input_file_paths[idx]).convert('RGB')
        img_target = Image.open(self.target_file_paths[idx]).convert('RGB')
        return img_input, img_target

    def __getitem__(self, idx):
        img_input, img_target = self.get_img_pair(idx)

        if self.is_train:
            img_input, img_target = get_patch(img_input, img_target)
            img_input, img_target = augment(img_input, img_target)

        img_input = rgb_to_tensor(img_input)
        img_target = rgb_to_tensor(img_target)

        return img_input, img_target

    def __len__(self):
        return self.n_samples
