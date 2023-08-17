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
from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

import sys
import os
from dataset import *
from model import *

# import kss

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

RANDOM_SEED = 42

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

print(pl.seed_everything(RANDOM_SEED))


'''
Data는 AI-HUB의 문서요약 텍스트의 json 파일을 사용한다.
'''

cwd = os.getcwd()
DATA_TRAIN_PATH = os.path.join(cwd, 'data', 'train_original_0.json')
# DATA_TRAIN_PATH = 'data/train_original_0.json'
df = pd.read_json(DATA_TRAIN_PATH)
df = df.dropna()
print('training data len: ', len(df))

DATA_TEST_PATH = os.path.join(cwd, 'data', 'valid_original_0.json')
# DATA_TEST_PATH = 'data/vaild_original_0.json'
test_df = pd.read_json(DATA_TEST_PATH)
test_df = test_df.dropna()
print('test data len: ', len(test_df))

# 데이터셋이 두개인 관계로 training set을 한 번 더 쪼개서 validation set을 만든다.
train_df, val_df = train_test_split(df, test_size=0.05)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
print('train/validation/test shape: ', train_df.shape, val_df.shape, test_df.shape)

# test setting으로 모든 데이터 다운사이즈
downsize = 2000
train_df = train_df[:downsize]
test_df = test_df[:downsize//10]
val_df = val_df[:downsize//10]
print('train/validation/test downsized: ', train_df.shape, val_df.shape, test_df.shape)

# dataframe preprocessing
print(train_df.head(1))
train_df = preprocess_data(train_df)
print(train_df.head(1))

# 본문 프린트해보기
i = 8
print('===== 본    문 =====')
for idx, str in enumerate(train_df['article_original'][i]):
    print(idx,':',str)
print('===== 요약정답 =====')
print(train_df['extractive'][i])
print('===== 추출본문 =====')
print('1 :', train_df['article_original'][i][train_df['extractive'][i][0]])
print('2 :', train_df['article_original'][i][train_df['extractive'][i][1]])
print('3 :', train_df['article_original'][i][train_df['extractive'][i][2]])
print('===== 생성본문 =====')
print(train_df['abstractive'][i])


'''
dataset
bert에서 여러문장을 입력하기 위해 presumm 에서 제안한 형식으로 인코딩 한다.

token embedding : < CLS > 문장 < SEP > 문장 < SEP > 문장 ... 문장 < SEP >
interval segment : 0 , 0 , 0 , 1 , 1 , 0 , 0 , ... 1 , 1
position embedding : 1 , 1 , 1 , 1 , 1 , 1 , 1 , ... 1 , 1
'''

'''
tokenizer
kpfBERT 토크나이저를 바로 쓴다. kpfBERT 토크나이저는 형태소와 유사하게 잘 토크나이징을 하게 설계되어 있다.
'''
BERT_MODEL_NAME = 'kpfbert'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

data_module = SummDataModule(
  train_df,
  test_df,  
  val_df,
  tokenizer,
  batch_size=BATCH_SIZE,
  max_token_len=MAX_TOKEN_COUNT
)

model = Summarizer(data_len=len(train_df))
print(model)

# # Load the TensorBoard extension
# os.system("python -m tensorboard.main")

# # Launch TensorBoard with the specified log directory
# logdir = "./lightning_logs"
# os.system(f"tensorboard --logdir {logdir} --bind_all")

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="avg_val_loss",
    mode="min"
)

logger = TensorBoardLogger("lightning_logs", name="kpfBERT_Summary")

early_stopping_callback = EarlyStopping(monitor='avg_val_loss', patience=3)

trainer = pl.Trainer(
    logger=logger,
    checkpoint_callback=checkpoint_callback,
    callbacks=[early_stopping_callback],
    max_epochs=N_EPOCHS,
    gpus=1,
    precision=16, # 소스 수정 또는 패키지 재설치 필요... 런타임 에러.
    progress_bar_refresh_rate=30
)

trainer.fit(model, data_module)
trainer.test()

