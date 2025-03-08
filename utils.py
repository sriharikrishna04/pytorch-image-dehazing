import os
import os.path
import torch
import sys
import numpy as np
from tensorboardX import SummaryWriter
from torchvision import transforms
import cv2

def PSNR(target, output):
    """Computes the Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse = np.mean((target - output) ** 2)
    if mse == 0:
        return 100  # Perfect match
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def tensor_to_rgb(x):
    """Convert tensor (HSV format) to RGB format for visualization."""
    output = x.detach().cpu().squeeze(0).numpy()  # Ensure tensor is on CPU
    output = output.transpose(1, 2, 0)  # Change shape from (C, H, W) -> (H, W, C)

    # Convert back to [0, 255] scale (assuming input is in range [-1, 1])
    output = (output * 255).clip(0, 255).astype(np.uint8)

    # Convert HSV to RGB
    output = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)

    return output


def rgb_to_tensor(x):
    """
    Convert an RGB image to an HSV tensor.
    """
    if isinstance(x, torch.Tensor):  # If input is a tensor, convert to NumPy
        x = x.permute(1, 2, 0).cpu().numpy()  # Convert (C, H, W) -> (H, W, C)
    
    if not isinstance(x, np.ndarray):  # Convert PIL image to NumPy
        x = np.array(x)

    hsv_image = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)  # Convert RGB to HSV
    hsv_tensor = torch.from_numpy(hsv_image).permute(2, 0, 1).float() / 255.0  # Normalize

    return hsv_tensor

def get_file_paths(imgdir):
    file_paths = []
    for file_name in os.listdir(imgdir):
        file_paths.append(os.path.join(imgdir, file_name))
    file_paths = sorted(file_paths)
    return file_paths

class SaveData:
    def __init__(self, save_dir, exp, finetuning):
        self.exp_dir = os.path.join(save_dir, exp)

        if not finetuning:
            if os.path.exists(self.exp_dir):
                os.system('rm -rf ' + self.exp_dir)
                print("! Remove a folder: " + self.exp_dir)

        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        self.model_dir = os.path.join(self.exp_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.logfile = open(self.exp_dir + '/logs.txt', 'a')

        tensorboard_dir = os.path.join(self.exp_dir, 'tb')
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        self.writer = SummaryWriter(tensorboard_dir)
        
    def get_latest_model_path(self):
        """Returns the latest checkpoint file path."""
        checkpoints = [f for f in os.listdir(self.model_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
        if not checkpoints:
            return None  # No checkpoint found
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        return os.path.join(self.model_dir, latest_checkpoint)

    def save_params(self, args):
        with open(self.exp_dir + '/params.txt', 'w') as params_file:
            params_file.write(str(args.__dict__) + "\n")

    def save_model(self,netG,netD,epoch,optimizerG,optimizerD,schedulerG,schedulerD):
        torch.save({'epoch': epoch,'netG': netG.state_dict(),'netD': netD.state_dict(),'optimizerG': optimizerG.state_dict(),'optimizerD': optimizerD.state_dict(),'schedulerG': schedulerG.state_dict(),'schedulerD': schedulerD.state_dict(),}, os.path.join(self.model_dir, f'checkpoint_epoch_{epoch}.pth'))

    def save_log(self, log):
        sys.stdout.flush()
        self.logfile.write(log + '\n')
        self.logfile.flush()

    def load_model(self, model):
        model.load_state_dict(torch.load(self.model_dir + '/model_lastest.pt'))
        last_epoch = torch.load(self.model_dir + '/last_epoch.pt')
        print("Load mode_status from {}/model_lastest.pt, epoch: {}".format(self.model_dir, last_epoch))
        return model, last_epoch

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, image):
        if self.pool_size == 0:
            return image
        if self.num_imgs < self.pool_size:
            self.images.append(image.clone())
            self.num_imgs += 1
            return image
        else:
            if np.random.uniform(0, 1) > 0.5:
                random_id = np.random.randint(self.pool_size, size=1)[0]
                tmp = self.images[random_id].clone()
                self.images[random_id] = image.clone()
                return tmp
            else:
                return image
