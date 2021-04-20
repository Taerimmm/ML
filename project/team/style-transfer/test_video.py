import os
import re
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchsummary

from PIL import Image

from test_config import *
from models import *
from utils import get_device, save_image

if __name__ == "__main__":
    device = get_device()

    with torch.no_grad():
        style_model = TransformerNet()

        ckpt_model_path = os.path.join(checkpoint_dir, checkpoint_file)
        checkpoint = torch.load(ckpt_model_path, map_location=device)

        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(checkpoint.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del checkpoint[k]

        style_model.load_state_dict(checkpoint['model_state_dict'])
        style_model.to(device)

        cap = cv2.VideoCapture(source_file)