import os
from PIL import Image

style_image_location = "03_style_image/SpongeBob.jpg"

# style_image_sample = Image.open(style_image_location, 'r')

batch_size = 8
random_seed = 10
num_epochs = 80
initial_lr = 1e-3
checkpoint_dir = "01_checkpoint/"

content_weight = 1e5
style_weight = 1e9 # default 1e10
log_interval = 50
checkpoint_interval = 100


transfer_learning = False
ckpt_model_path = os.path.join(checkpoint_dir, "VGG19_MaxPool_SpongeBob_ckpt_epoch_87.pth")
