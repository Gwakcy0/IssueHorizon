# 참고 링크: https://github.com/KPFBERT/kpfbertsum/blob/main/kpfbert_summary.ipynb
# 사용 데이터: ETRI의 AI-HUB에서 제공하는 문서요약 텍스트(비플라이소프트) 
# -> https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=97

'''
전체기사에서 중요한 순서대로 상위 3개의 문장을 추출해서 제시하는 기사 요약 서비스이다.

pytorch-lightning을 이용하여 전체프로세서를 작성하였다.

BERT를 이용한 SUMMARY 관련 논문 및 nlpyang의 PreSumm 소스를 참조하였다.
'''
import math
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from torch.nn.init import xavier_uniform_

import pytorch_lightning as pl
# from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

# import kss

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

RANDOM_SEED = 42

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

print(pl.seed_everything(RANDOM_SEED))

MAX_TOKEN_COUNT = 512
N_EPOCHS = 10
BATCH_SIZE = 4


'''
Data: AI-HUB의 문서요약 텍스트의 json 파일을 사용한다. 


'''
DATA_TRAIN_PATH = 'data/train_original.json'
df = pd.read_json(DATA_TRAIN_PATH)
df = df.dropna()
len(df)
