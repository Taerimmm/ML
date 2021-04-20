import os
import re
import torch
import torchvision.transforms as transforms

from models import TransformerNet
from utils import load_image, save_image, get_device
from test_config import *

if __name__ == "__main__":
    device = get_device()

    content_image = load_image(test_image)

    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()

        ckpt_model_path = os.path.join(checkpoint_dir, checkpoint_file)
        checkpoint = torch.load(ckpt_model_path, map_location=device)

        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(checkpoint.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k): # in200.running_var or in200.running_mean
                del checkpoint[k]

        style_model.load_state_dict(checkpoint['model_state_dict'])
        style_model.to(device)

        output = style_model(content_image).cpu()

    save_image(output_image, output[0])
    print('Save image !!')
