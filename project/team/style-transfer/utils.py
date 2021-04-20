import os
from PIL import Image
import numpy as np

def get_device():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device :', device)
    return device

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if filename[-3:] == 'png':
        img = Image.open(filename).convert('RGB')

    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img

def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)

# 참고 : https://programmersought.com/article/73943029361/
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)  # w and h together into vector form (행렬을 벡터로 표현 가능하기 때문에)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)  # 행렬 곱  , / (ch * h * w) : 0 ~ 1 전처리
    return gram

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)  # torch.Size([3, 1, 1])
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)   # torch.Size([3, 1, 1])
    batch = batch.div_(255.0)
    return (batch - mean) / std