import numpy as np
import pandas as pd
import glob
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MnistDataset_v2(Dataset):
    def __init__(self, imgs=None, labels=None, transform=None, train=True):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.train = train
        pass

    def __len__(self):
        # 데이터 총 샘플 수 
        return len(self.imgs)

    def __getitem__(self, idx):
        # 1개 셈플 get
        img = self.imgs[idx]
        img = self.transform(img)
        
        if self.train == True:
            label = self.labels[idx]
            return img, label
        else:
            return img 
            
class EfficientNet_MultiLabel(nn.Module):
    def __init__(self, in_channels):
        super(EfficientNet_MultiLabel, self).__init__()
        self.network = EfficientNet.from_pretrained('efficientnet-b0', in_channels=in_channels)
        self.output_layer = nn.Linear(1000, 26)

    def forward(self, x):
        x = F.relu(self.network(x))
        x = torch.sigmoid(self.output_layer(x))
        return x

## Test Image 로드
test_imgs_dir = np.array(sorted(glob.glob('../data/test_dirty_mnist_2nd/*')))


test_imgs = []
for path in tqdm(test_imgs_dir):
    test_img = cv2.imread(path, cv2.IMREAD_COLOR)
    test_imgs.append(test_img)
test_imgs = np.array(test_imgs)

test_transform = transforms.Compose([
    transforms.ToTensor(),
])


## Test 추론


submission = pd.read_csv('./dacon3/data/sample_submission.csv')

with torch.no_grad():
    for fold in range(5):
        model = EfficientNet_MultiLabel(in_channels=3).to(device)
        model.load_state_dict(torch.load('./dacon3/data/EfficientNetB0-fold{}.pt'.format(fold)))
        model.eval()

        test_dataset = MnistDataset_v2(imgs=test_imgs, transform=test_transform, train=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

        for n, X_test in enumerate(tqdm(test_loader)):
            X_test = torch.tensor(X_test, device=device, dtype=torch.float32)
            with torch.no_grad():
                model.eval()
                pred_test = model(X_test).cpu().detach().numpy()
                submission.iloc[n*32:(n+1)*32, 1:] += pred_test / 5


## 제출물 생성


submission.iloc[:,1:] = np.where(submission.values[:,1:] >= 0.5, 1, 0)

submission.to_csv('./dacon3/data/EfficientNetB0-fold0.csv', index=False)
