# CUDA_VISIBLE_DEVICES=1 nohup python -u D_MI_FGSM_2.py --port 7200 > D_MI_FGSM_2_TestResults.out 2>&1 &
setup = "[ NO Normalization | step: alpha=epsilon/iter | type: D-MI-FGSM ]"
print(f'Training SETUP -> {setup}')
print('(Delta applied ONLY at Diversity Function Call. NOT at RESNET Model Inference)')

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
# max_iterations = [50, 100, 200, 400, 800, 1600]
max_iterations = [1600]
input_path = 'images/'
epochs = int(np.ceil(len(ids) / batch_size))

img_size = 299

epsilons = [48/255, 32/255, 16/255]

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

def add_input_diversity(X_ori, resize_rate=0.9, diversity_factor=0.5):  # 랜덤하게 input image를 transform해서 다양한 모델들에 대한 transferability를 확보하는 목적 (DI-FGSM의 원리)
    if torch.rand(1) < diversity_factor:
        return X_ori

    # calculate the original image size and the resized image size based on the provided resize_rate
    img_size = X_ori.shape[-1]
    img_resize = int(img_size * resize_rate)

    if resize_rate < 1:
        img_size = img_resize
        img_resize = X_ori.shape[-1]
    
    # generate random integer(with value ranging between img_size and image_resize) and rescale the input images using bilinear interpolation to the size [rnd, rnd]
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(X_ori, size=[rnd, rnd], mode='nearest')

    # calculate the remaining height 'h_rem' and remaining width 'w_rem' after rescaling
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd

    # generate random padding values for the top, bottom, left, and right sides within the calculated remaining height and width
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left

    # pad the rescaled images with zeros, based on the calculated padding values
    pad = [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()]
    padded = F.pad(rescaled, pad=pad, mode='constant', value=0)

    # return the diversified batch of images after applying input diversity through resizing and padding
    return padded

print('<<BEGIN>>')
print('='*50)

sota_performance = 0
sota_setup = {'max_iteration':0, 'epsilon':0}

for epsilon in epsilons:
    for max_iteration in max_iterations:
        preds_ls_vgg = []
        preds_ls_resnet = []
        preds_ls_inception = []
        labels_ls =[]
        origin_ls = []
        torch.cuda.empty_cache()
        for k in tqdm(range(epochs), total=epochs):  # 에포크 만큼
            batch_size_cur = min(batch_size, len(ids) - k * batch_size)  # 전체 데이터 수와 에포크 사이즈 이용해 배치사이즈 계산
            X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)  # 원본 이미지 저장하기 위한 텐서 정의
            delta = torch.zeros_like(X_ori, requires_grad=True).to(device)  # 노이즈 저장하기 위한 텐서 정의
            momentum = torch.zeros_like(X_ori).to(device)  # 0으로 초기화된 동일차원의 모멘텀 텐서 정의

            for i in range(batch_size_cur):  # 배치 사이즈 갯수만큼 이미지 갖고와서
                X_ori[i] = trn(Image.open(input_path + ids[k * batch_size + i] + '.png'))  # 원본 이미지 저장하는 텐서에 원본 이미지값 저장
            ori_idx = origins[k * batch_size:k * batch_size + batch_size_cur]  # 원본 이미지의 실제 클래스 라벨 저장
            labels = torch.tensor(targets[k * batch_size:k * batch_size + batch_size_cur]).to(device)  # 델타를 학습시킬 때 사용할 가짜 라벨 저장
            prev = float('inf')  # 어따쓰는겨

            alpha = epsilon/max_iteration
            for _ in range(max_iteration):  # I-FGSM
                X_ori = add_input_diversity(X_ori + delta, 0.9, 0.5)  # DI-FGSM
                logits = resnet(norm(X_ori))  # 노이즈 추가한 이미지를 ResNet이 분류한 분산표상
                loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)  # 위 분산표상과 "가짜라벨"을 비교해 로스 계산
                loss.backward()  # 역전파
                grad_c = delta.grad.clone()  # 딥카피로 그래디언트 계산한 값 따로 저장해두기 (아래서 0으로 다시 리셋하기 위해)
                delta.grad.zero_()  # 그래디언트 속성값 0으로 초기화 (걍 파이토치가 이걸 안 해줘서 하나봄)
                momentum = decay * momentum + grad_c / torch.norm(grad_c, p=1)
              
                delta.data = delta.data - alpha * torch.sign(momentum)  # lr은 하이퍼파라미터
                delta.data = delta.data.clamp(-epsilon, epsilon)  # 지나치게 큰 폭의 델타값 업데이트 방지 위한 clipping으로 이해
                delta.data = ((X_ori + delta.data).clamp(0,1)) - X_ori  # ex) (0~1 + 16/255).clamp(0,1) - 0~1 = 0~1  (0보다 작거나 1보다 큰 이상치 값을 갖는 픽셀값을 클램핑)

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

        epsilon_rounded = round(epsilon, 2)
        print(f'| Iter: {max_iteration} | Eps: {epsilon_rounded} |')


      
        match_percentage_vgg = 100 * accuracy_score(df['label'], df['pred_vgg'])
        if match_percentage_vgg > sota_performance:
            sota_performance = match_percentage_vgg
            sota_setup['max_iteration'] = max_iteration
            sota_setup['epsilon'] = epsilon
            print(f'Targeted Attack Score (VGG16): {match_percentage_vgg} % <------------------VGG16 TEST SOTA')
        else:
            print(f'Targeted Attack Score (VGG16): {match_percentage_vgg} %')
        mismatch_percentage_vgg = 100 * (df['origin'] != df['pred_vgg']).mean()
        print(f'Untargeted Attack Score (VGG16): {mismatch_percentage_vgg} %')


        match_percentage_inception = 100 * accuracy_score(df['label'], df['pred_inception'])
        print(f'Targeted Attack Score (InceptionV3): {match_percentage_inception} %')
        mismatch_percentage_inception = 100 * (df['origin'] != df['pred_inception']).mean()
        print(f'Untargeted Attack Score (InceptionV3): {mismatch_percentage_inception} %')

        match_percentage_resnet = 100 * accuracy_score(df['label'], df['pred_resnet'])
        print(f'Targeted Attack Score (ResNet): {match_percentage_resnet} %')
        mismatch_percentage_resnet = 100 * (df['origin'] != df['pred_resnet']).mean()
        print(f'Untargeted Attack Score (ResNet): {mismatch_percentage_resnet} %')

        df_submission = df[['origin', 'pred_vgg', 'label']].copy()
        df_submission.rename(columns={'pred_vgg': 'pred'}, inplace=True)
        df_submission.to_csv(f'submission_iter{max_iteration}_eps{epsilon_rounded}_1Delta.csv')
        print(f'submission.csv file for iter{max_iteration}_eps{epsilon_rounded}_1Delta SAVED SUCCESSFULLY')
        
        print('='*50)
        
print('<<END>>')