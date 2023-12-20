# python log_interpreter.py

import re
import numpy as np
import pandas as pd
import tqdm
import csv

def reshape_2d(original_list):
  num_rows = len(original_list) // 3
  num_columns = 3

  array_data = np.array(original_list[:num_rows * num_columns], dtype=float).reshape(num_rows, num_columns)
  if num_rows == 12:
    index = ['5', '10', '25', '50', '75', '100', '150', '200', '250', '300', '350', '400']
    columns=['0.06', '0.12', '0.18']
  elif num_rows == 9:
    index = ['5', '10', '25', '50', '75', '100', '150', '200', '250']
    columns=['0.06', '0.12', '0.18']
  elif num_rows == 7:
    index = ['5', '10', '25', '50', '75', '100', '200']
    columns=['0.06', '0.12', '0.18']
  elif num_rows == 6:
    index = ['50', '100', '200', '400', '800', '1600']
    columns=['0.18', '0.12', '0.06']
  elif num_rows == 5:
    index = ['50', '100', '200', '400', '800']
    columns=['0.18', '0.12', '0.06']
  elif num_rows == 1:
    index = ['1600']
    columns=['0.18', '0.12', '0.06']
  else:
    print('index length mismatch!!!!!!!')
  df = pd.DataFrame(array_data, columns=columns, index=index)
  # print(df)
  return df

file_paths = [
    'project_6_testResults.out',
    'project_7_testResults.out',
    'project_8_testResults.out',
    'project_9_testResults.out',
    # 'D_MI_FGSM_1_TestResults.out',
    # 'D_MI_FGSM_2_TestResults.out'
    ]

file_names = [
    'MI_FGSM_TestResults.csv',
    'D-MI_FGSM_TestResults.csv',
    'N-MI_FGSM_TestResults.csv',
    'T-MI_FGSM_TestResults.csv',
    # 'D-MI_FGSM_TestResults_Final_1.xlsx',
    # 'D-MI_FGSM_TestResults_Final_2.xlsx'
]

for i in range(len(file_paths)):
    file = file_paths[i]
    with open(file, 'r') as file:
        log_data = file.read()

    targeted_vgg_pattern = re.compile(r'Targeted Attack Score \(VGG16\): ([\d.]+) %.*?')
    untargeted_vgg_pattern = re.compile(r'Untargeted Attack Score \(VGG16\): ([\d.]+) %.*?')
    # targeted_inception_pattern = re.compile(r'Targeted Attack Score \(InceptionV3\): ([\d.]+) %.*?')
    # untargeted_inception_pattern = re.compile(r'Untargeted Attack Score \(InceptionV3\): ([\d.]+) %.*?')
    # targeted_resnet_pattern = re.compile(r'Targeted Attack Score \(ResNet\): ([\d.]+) %.*?')
    # untargeted_resnet_pattern = re.compile(r'Untargeted Attack Score \(ResNet\): ([\d.]+) %.*?')

    targeted_vgg_scores = targeted_vgg_pattern.findall(log_data)
    untargeted_vgg_scores = untargeted_vgg_pattern.findall(log_data)
    # targeted_inception_scores = targeted_inception_pattern.findall(log_data)
    # untargeted_inception_scores = untargeted_inception_pattern.findall(log_data)
    # targeted_resnet_scores = targeted_resnet_pattern.findall(log_data)
    # untargeted_resnet_scores = untargeted_resnet_pattern.findall(log_data)

    # print(targeted_vgg_scores)
    # print(untargeted_vgg_scores)

    reshape_2d(targeted_vgg_scores).to_csv('Targeted_'+file_names[i], index=True)
    reshape_2d(untargeted_vgg_scores).to_csv('Untargeted_'+file_names[i], index=True)