'''
D-MI-FGSM (Momentum Iterative FGSM)
'''

# CUDA_VISIBLE_DEVICES=0 nohup python -u project_4.py --port 6100 > project_4_testResults.out 2>&1 &

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
        '''return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]''' # 
        return x  # Modification: No Normalization!

norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trn = transforms.Compose([transforms.ToTensor(),])
ids, origins, targets, gt = load_ground_truth('train.csv')
batch_size = 20
max_iterations = [5, 10, 20, 40] #, 80, 160, 320, 640]
input_path = 'images/'
epochs = int(np.ceil(len(ids) / batch_size))

img_size = 299
lrs = [2/255, 4/255, 8/255, 16/255, 32/255, 64/255, 128/255] #step size

epsilons = [16/255, 32/255, 48/255, 64/255, 80/255, 96/255, 112/255, 128/255] # L_inf norm bound

decay = 0.9

resnet = models.resnet50(weights="IMAGENET1K_V1").eval()
vgg = models.vgg16_bn(weights="IMAGENET1K_V1").eval()
inception_v3 = models.inception_v3(weights="IMAGENET1K_V1").eval()

for param in resnet.parameters():
    param.requires_grad = False
for param in vgg.parameters():
    param.requires_grad = False

resnet.to(device)
vgg.to(device)
inception_v3.to(device)

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

'''
D-MI-FGSM
'''

def add_input_diversity(X_ori, diversity_factor=0.1):  # 랜덤하게 input image를 transform해서 다양한 모델들에 대한 transferability를 확보하는 목적 (DI-FGSM의 원리)
  noise = torch.randn_like(X_ori).to(device) * diversity_factor
  X_ori = X_ori + noise
  '''X_ori = torch.clamp(X_ori, 0, 1)'''
  return X_ori

print('='*100)
print('BEGIN')
print('='*100)

sota_performance = 0
sota_setup = {'max_iteration':0, 'lr':0, 'epsilon':0}

# for max_iteration in max_iterations:
len_of_max_iterations = len(max_iterations)
len_of_lrs = len(lrs)
len_of_epsilons = len(epsilons)

for max_iteration in max_iterations:
# for cnt_iteration in tqdm(range(len_of_max_iterations), total=len_of_max_iterations):
  # max_iteration = max_iterations[cnt_iteration]
  for lr in lrs:
  # for cnt_lr in tqdm(range(len_of_lrs), total=len_of_lrs):
    # lr = lrs[cnt_lr]
    for epsilon in epsilons:
    # for cnt_epsilon in tqdm(range(len_of_epsilons), total=len_of_epsilons):
      # epsilon = epsilons[cnt_epsilon]
      
      preds_ls_vgg = []
      preds_ls_resnet = []
      preds_ls_inception = []
      labels_ls =[]
      origin_ls = []

      torch.cuda.empty_cache()
      # for k in tqdm(range(epochs), total=epochs):  # 에포크 만큼
      for k in range(epochs):  # 에포크 만큼
          batch_size_cur = min(batch_size, len(ids) - k * batch_size)  # 전체 데이터 수와 에포크 사이즈 이용해 배치사이즈 계산
          X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)  # 원본 이미지 저장하기 위한 텐서 정의
          delta = torch.zeros_like(X_ori, requires_grad=True).to(device)  # 노이즈 저장하기 위한 텐서 정의
          momentum = torch.zeros_like(X_ori).to(device)  # 0으로 초기화된 동일차원의 모멘텀 텐서 정의
          delta_optimizer = optim.Adam([delta], lr=lr)  # 수렴 도와주는 옵티마이저 정의

          for i in range(batch_size_cur):  # 배치 사이즈 갯수만큼 이미지 갖고와서
              X_ori[i] = trn(Image.open(input_path + ids[k * batch_size + i] + '.png'))  # 원본 이미지 저장하는 텐서에 원본 이미지값 저장
          ori_idx = origins[k * batch_size:k * batch_size + batch_size_cur]  # 원본 이미지의 실제 클래스 라벨 저장
          labels = torch.tensor(targets[k * batch_size:k * batch_size + batch_size_cur]).to(device)  # 델타를 학습시킬 때 사용할 가짜 라벨 저장
          prev = float('inf')  # 어따쓰는겨
          for t in range(max_iteration):  # I-FGSM
              '''X_ori = add_input_diversity(X_ori)'''  # D-MI-FGSM
              logits = resnet(norm(X_ori + delta))  # 노이즈 추가한 이미지를 ResNet이 분류한 분산표상
              loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)  # 위 분산표상과 "가짜라벨"을 비교해 로스 계산
              loss.backward()  # 역전파
              grad_c = delta.grad.clone()  # 딥카피로 그래디언트 계산한 값 따로 저장해두기 (아래서 0으로 다시 리셋하기 위해)
              delta.grad.zero_()  # 그래디언트 속성값 0으로 초기화 (걍 파이토치가 이걸 안 해줘서 하나봄)
              '''modification : Momentum Iterative FGSM update'''
              momentum = decay * momentum + grad_c / torch.norm(grad_c, p=1)

              delta.data = delta.data - lr * torch.sign(momentum)  # lr은 하이퍼파라미터
              delta.data = delta.data.clamp(-epsilon, epsilon)  # 지나치게 큰 폭의 델타값 업데이트 방지 위한 clipping으로 이해
              delta.data = ((X_ori + delta.data).clamp(0,1)) - X_ori  # ex) (0~1 + 16/255).clamp(0,1) - 0~1 = 0~1  (0보다 작거나 1보다 큰 이상치 값을 갖는 픽셀값을 클램핑)

              delta_optimizer.step()

          # t번 돌고 나온 최종 델타로 테스트용 모델들에 넣어서 출력 뽑기
          X_pur = norm(X_ori + delta)

          preds_vgg = torch.argmax(vgg(X_pur), dim=1)
          preds_resnet = torch.argmax(resnet(X_pur), dim=1)
          preds_inception = torch.argmax(inception_v3(X_pur), dim=1)

          preds_ls_vgg.append(preds_vgg.cpu().numpy())
          preds_ls_resnet.append(preds_resnet.cpu().numpy())
          preds_ls_inception.append(preds_inception.cpu().numpy())

          labels_ls.append(labels.cpu().numpy())
          origin_ls.append(ori_idx)


      df = pd.DataFrame({
          'pred_vgg': [a for b in preds_ls_vgg for a in b],
          'pred_resnet': [a for b in preds_ls_resnet for a in b],
          'pred_inception': [a for b in preds_ls_inception for a in b],
          'origin': [a for b in origin_ls for a in b],
          'label': [a for b in labels_ls for a in b]
      })

      
      print(f'| Iter: {max_iteration} | LR: {round(lr, 5)} | Eps: {epsilon} |')


      
      match_percentage_vgg = 100 * accuracy_score(df['label'], df['pred_vgg'])
      if match_percentage_vgg > sota_performance:
        print("+"*40 + "VGG16 TEST SOTA PERFORMANCE" + "+"*40)
        sota_performance = match_percentage_vgg
        sota_setup['max_iteration'] = max_iteration
        sota_setup['lr'] = round(lr, 5)
        sota_setup['epsilon'] = epsilon
      print(f'With-Target Score (VGG16): {match_percentage_vgg}')
      
      mismatch_percentage_vgg = 100 * (df['origin'] != df['pred_vgg']).mean()
      print(f'No-Target Score (VGG16): {mismatch_percentage_vgg}')


      
      match_percentage_inception = 100 * accuracy_score(df['label'], df['pred_inception'])
      print(f'With-Target Score (InceptionV3): {match_percentage_inception}')

      mismatch_percentage_inception = 100 * (df['origin'] != df['pred_inception']).mean()
      print(f'No-Target Score (InceptionV3): {mismatch_percentage_inception}')


      
      match_percentage_resnet = 100 * accuracy_score(df['label'], df['pred_resnet'])
      print(f'With-Target Score (ResNet): {match_percentage_resnet}')

      mismatch_percentage_resnet = 100 * (df['origin'] != df['pred_resnet']).mean()
      print(f'No-Target Score (ResNet): {mismatch_percentage_resnet}')
      
      

      print('#'*50)
