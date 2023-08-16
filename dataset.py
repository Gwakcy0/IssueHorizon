import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pytorch_lightning as pl

MAX_TOKEN_COUNT = 512
N_EPOCHS = 10
BATCH_SIZE = 4

# json 파일 혹은 딕셔너리 데이터를 dataframe으로 바꾸기
def preprocess_data(data):
    outs = []
    for doc in data['documents']:
        line = []
        line.append(doc.get('media_name'))
        line.append(doc['id'])
        para = []
        for sent in doc['text']:
            for s in sent:
                para.append(s['sentence'])
        line.append(para)
        line.append(doc['abstractive'][0])
        line.append(doc['extractive'])
        a = doc['extractive']
        if a[0] == None or a[1] == None or a[2] == None:
            continue
        outs.append(line)

    outs_df = pd.DataFrame(outs)
    outs_df.columns = ['media', 'id', 'article_original', 'abstractive', 'extractive']
    return outs_df

class SummDataset(Dataset):

    def __init__(
        self, 
        data: pd.DataFrame, 
        tokenizer: BertTokenizer, 
        max_token_len: int = 512
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        tokenlist = []
        for sent in data_row.article_original:
            tokenlist.append(self.tokenizer(
                text = sent,
                add_special_tokens = True)) #, # Add '[CLS]' and '[SEP]'
    
        src = [] # 토크나이징 된 전체 문단
        labels = []  # 요약문에 해당하면 1, 아니면 0으로 문장수 만큼 생성
        segs = []  #각 토큰에 대해 홀수번째 문장이면 0, 짝수번째 문장이면 1을 매핑
        clss = []  #[CLS]토큰의 포지션값을 지정

        odd = 0
        for tkns in tokenlist:
            if odd > 1 : odd = 0
            clss = clss + [len(src)]
            src = src + tkns['input_ids']
            segs = segs + [odd] * len(tkns['input_ids'])
            if tokenlist.index(tkns) in data_row.extractive :
                labels = labels + [1]
            else:
                labels = labels + [0]
            odd += 1
        
            #truncation
            if len(src) == MAX_TOKEN_COUNT:
                break
            elif len(src) > MAX_TOKEN_COUNT:
                src = src[:self.max_token_len - 1] + [src[-1]]
                segs = segs[:self.max_token_len]
                break
    
        #padding
        if len(src) < MAX_TOKEN_COUNT:
            src = src + [0]*(self.max_token_len - len(src))
            segs = segs + [0]*(self.max_token_len - len(segs))
            
        if len(clss) < MAX_TOKEN_COUNT:
            clss = clss + [-1]*(self.max_token_len - len(clss))
        if len(labels) < MAX_TOKEN_COUNT:
            labels = labels + [0]*(self.max_token_len - len(labels))

        return dict(
            src = torch.tensor(src),
            segs = torch.tensor(segs),
            clss = torch.tensor(clss),
            labels= torch.FloatTensor(labels)
        )

class SummDataModule(pl.LightningDataModule):

    def __init__(self, train_df, test_df, val_df, tokenizer, batch_size=1, max_token_len=512):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = SummDataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len
        )

        self.test_dataset = SummDataset(
            self.test_df,
            self.tokenizer,
            self.max_token_len
        )
    
        self.val_dataset = SummDataset(
            self.val_df,
            self.tokenizer,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0 # windows는 0으로 고정해야 에러 안난다. num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=0 # windows는 0으로 고정해야 에러 안난다. num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0 # windows는 0으로 고정해야 에러 안난다. num_workers=2
        )