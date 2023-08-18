import os
import pandas as pd
import warnings
from text_preprocessing import *

def load_dataframes():
    path = os.getcwd()
    path = os.path.join(path, 'data')
    print(f'PATH: {path}')

    for dir in os.listdir(path):
        p = os.path.join(path, dir)
        if os.path.isdir(p) and dir != 'docSum_texts':
            print(f'========== Current dataset: {dir} ==========')
            for file in os.listdir(p):
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    yield dir, pd.read_excel(os.path.join(p, file), sheet_name=None, header=1, dtype={'작성일': str, '작성자': str, '제목': str, '내용': str, '댓글수': int})

df_list = []
for dir, dic in load_dataframes():
    for media, df in dic.items():
        if media == '뉴스':
            continue
        length = df.shape[0]
        df.insert(0, '매체유형', [media] * length)
        df.insert(0, '키워드', [dir] * length)
        df.insert(2, '개수', [1] * length)
        df_list.append(clean_dataset(df))
print('Concatenating...')
final_df = pd.concat(df_list, ignore_index=True)
final_df.to_csv('./data/all.csv', mode = 'w')

# issueMedia = {}
# for dir, df in load_dataframes():
#     for media in df.keys():
#         if (dir, media) not in issueMedia:
#             issueMedia[(dir, media)] = df[media]
#         else:
#             df_temp = issueMedia[(dir, media)]
#             issueMedia[(dir, media)] = pd.concat([df_temp, df[media]], ignore_index=True)

# for (dir, media), df in issueMedia.items():
#     print(dir, media)
#     # df = clean_dataset(df)
#     df.to_csv(f'./data/clean_{dir}_{media}.csv', mode = 'w')

# df = pd.read_csv('./data/all.csv')
# print(df.info())

# # 2. Make Clean Korean News Corpus for pre-training
# # 2-1. Load all cleaned dataframe
# def load_csv():
#     path = os.getcwd()
#     path = os.path.join(path, 'data')
#     print(f'PATH: {path}')

#     for file in os.listdir(path):
#         if file[-4:] == '.csv':
#             print(f'loading file {file} ...')
#             p = os.path.join(path, file)
#             yield pd.read_csv(p)

# total_content = []
# for df in load_csv():
#     print(df.head())
#     df['내용'] = df['내용'].to_list()
#     total_content.extend(df['내용'])

# print("\n >> Total Content size : ", len(total_content))

# # 2-3. Make clean News Corpus for pretraining
# issue_corpus = []
# for news in total_content:
#     if type(news) is str:
#         issue_sentences = news.split(".")
#         for sent in issue_sentences:
#             issue_corpus.append(sentence_split(sent))

# issue_corpus = [x for x in issue_corpus if x != 'None']

# print("\n >> Issue Corpus size : ", len(issue_corpus))

# # 2-4. List to a File line by line
# with open('./data/korean_issue_corpus.txt', 'w') as fp:
#     for row in issue_corpus:
#         fp.write("%s\n" % row)
#     print("Write is Done.")