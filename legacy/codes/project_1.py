# CUDA_VISIBLE_DEVICES=0 python project_1.py

import os
import math
import csv
import pickle
from urllib import request
import scipy.stats as st

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

device = torch.device("cuda:0")

##load image metadata (Image_ID, true label, and target label)
def load_ground_truth(fname):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    df = pd.read_csv(fname)
    for _, row in df.iterrows():
        image_id_list.append( row['ImageId'] )
        label_ori_list.append( int(row['TrueLabel']) - 1 )
        label_tar_list.append( int(row['TargetClass']) - 1 )
    gt = pickle.load(request.urlopen('https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'))
    return image_id_list,label_ori_list,label_tar_list, gt

## simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

"""# Training"""

norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trn = transforms.Compose([transforms.ToTensor(),])
ids, origins, targets, gt = load_ground_truth('train.csv')

batch_size = 20
max_iterations = 100
input_path = 'images/'
epochs = int(np.ceil(len(ids) / batch_size))

img_size = 299
lr = 2 / 255 #step size
epsilon = 16 # L_inf norm bound

resnet = models.resnet50(weights="IMAGENET1K_V1").eval()
vgg = models.vgg16_bn(weights="IMAGENET1K_V1").eval()

for param in resnet.parameters():
    param.requires_grad = False
for param in vgg.parameters():
    param.requires_grad = False

resnet.to(device)
vgg.to(device)

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

preds_ls = []
labels_ls =[]
origin_ls = []

torch.cuda.empty_cache()
for k in tqdm(range(epochs), total=epochs):  # 에포크 만큼
    batch_size_cur = min(batch_size, len(ids) - k * batch_size)
    X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)  # 원본 이미지 저장하기 위한 텐서 정의
    delta = torch.zeros_like(X_ori, requires_grad=True).to(device)  # 노이즈 저장하기 위한 텐서 정의
    for i in range(batch_size_cur):  # 배치 사이즈 갯수만큼 이미지 갖고와서
        X_ori[i] = trn(Image.open(input_path + ids[k * batch_size + i] + '.png'))  # 원본 이미지 저장하는 텐서에 원본 이미지값 저장
    ori_idx = origins[k * batch_size:k * batch_size + batch_size_cur]  # 원본 이미지의 실제 클래스 라벨 저장
    labels = torch.tensor(targets[k * batch_size:k * batch_size + batch_size_cur]).to(device)  # 델타를 학습시킬 때 사용할 가짜 라벨 저장

    prev = float('inf')
    for t in range(max_iterations):  # 이건 I-FGSM 같은데...........  ㄴㄴ 아닌듯 다시 보니까 max_iterations값 상관없이 1번 돈거랑 똑같은 delta가 나옴
        logits = resnet(norm(X_ori + delta))  # 노이즈 추가한 이미지를 ResNet이 분류한 분산표상
        loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)  # 위 분산표상과 "가짜라벨"을 비교해 로스 계산
        loss.backward()  # 역전파
        grad_c = delta.grad.clone()  # 딥카피로 그래디언트 계산한 값 따로 저장해두기 (아래서 0으로 다시 리셋하기 위해)
        delta.grad.zero_()  # 그래디언트 속성값 0으로 초기화 (걍 파이토치가 이걸 안 해줘서 하나봄)
        delta.data = delta.data - lr * torch.sign(grad_c)  # 가짜 라벨에 대한 로스 최소화되도록 델타값 업데이트
        delta.data = delta.data.clamp(-epsilon / 255, epsilon / 255)  # 지나치게 큰 폭의 델타값 업데이트 방지 위한 clipping으로 이해
        delta.data = ((X_ori + delta.data).clamp(0,1)) - X_ori  # ex) (150 + 16/255).clamp(0,1) - 150 = -149
        # delta.data = ((X_ori + delta.data).clamp(0,255)) - X_ori  # ㅇㄴ 근데 이걸 왜 0~1로 클램핑해 0~255로 하는거면 노이즈 더한 뒤에 음수나 255보다 큰 값 갖는 노이즈 생길까봐 그런거라 치는데,,,,
    '''
    for t in range(max_iterations):  # 이건 I-FGSM 같은데...........  ㄴㄴ 아닌듯 다시 보니까 max_iterations값 상관없이 1번 돈거랑 똑같은 delta가 나옴
        logits = resnet(norm(X_ori + delta))  # 노이즈 추가한 이미지를 ResNet이 분류한 분산표상
        loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)  # 위 분산표상과 "가짜라벨"을 비교해 로스 계산
        loss.backward()  # 역전파
        grad_c = delta.grad.clone()  # 딥카피로 그래디언트 계산한 값 따로 저장해두기 (아래서 0으로 다시 리셋하기 위해)
        delta.grad.zero_()  # 그래디언트 속성값 0으로 초기화 (걍 파이토치가 이걸 안 해줘서 하나봄)
        delta.data = delta.data - lr * torch.sign(grad_c)  # 가짜 라벨에 대한 로스 최소화되도록 델타값 업데이트
        delta.data = delta.data.clamp(-epsilon / 255,epsilon / 255)  # 지나치게 큰 폭의 델타값 업데이트 방지 위한 clipping으로 이해
        delta.data = ((X_ori + delta.data).clamp(0,1)) - X_ori  # 
    '''
    # t번 돌고 나온 최종 델타로 테스트용 모델인 VGG에 넣어서 출력 뽑기
    X_pur = norm(X_ori + delta)
    preds = torch.argmax(vgg(X_pur), dim=1)

    preds_ls.append(preds.cpu().numpy())
    labels_ls.append(labels.cpu().numpy())
    origin_ls.append(ori_idx)

"""# Evaluation
* **Save the created data frame as csv, download and submit it**
* Explore submission file by clicking on the 'Folder' shaped icon on the left tab of the Colab
* Right-click the submission.csv item and download it

<div>
<img src="https://www.notion.so/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F79c40b22-24a8-4f42-b27c-09cb5cb721ca%2Fd1aa9a27-80c9-4ef3-9019-8518e2d1e564%2F%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-12-04_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_6.56.55.png?table=block&id=5b8c7e1a-9710-42c3-b8d4-da8eebca66e5&spaceId=79c40b22-24a8-4f42-b27c-09cb5cb721ca&width=2000&userId=9b171d95-14c1-4887-82b9-c7e38516fa74&cache=v2" width="500"/>
</div>


"""

df = pd.DataFrame({
    'origin': [a for b in origin_ls for a in b],
    'pred': [a for b in preds_ls for a in b],
    'label': [a for b in labels_ls for a in b]
})

df.head()

# accuracy_score(df['label'], df['pred'])
# print(accuracy_score(df['label'], df['pred']))
match_percentage = 100 * accuracy_score(df['label'], df['pred'])
print(f'조교님이 설정한 방식대로 계산한 정확도 점수: {match_percentage}')

mismatch_percentage = 100 * (df['origin'] != df['pred']).mean()
print(f'공격 성공 여부만 따져서 계산한 정확도 점수: {mismatch_percentage}')

"""* This performance will not be reproduced with Colab. Please don't worry and do your best.

"""

df.to_csv('submission.csv')

'''
"""# Visualization"""

def viz(img_A, img_B, origins, labels, gt, preds):
    for img_a, img_b, origin, label, pred in zip(img_A, img_B, origins, labels, preds):
        img_a = img_a.permute(1, 2, 0)
        img_b = img_b.permute(1, 2, 0)

        fig, (axA, axB) = plt.subplots(1, 2, figsize=(10,3))
        axA.imshow(img_a)
        axA.set_title("True label: " + gt[origin])
        axB.imshow(img_b)
        axB.set_title("Target: " + gt[label])

        result = 'Failed' if pred != label else 'Success'
        caption = f'Pred: {gt[pred]} -> {result}'
        fig.text(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=12)

        plt.show()

viz(X_ori.cpu().detach(), X_pur.cpu().detach(), ori_idx, labels.cpu().numpy(), gt, preds.cpu().numpy())
'''