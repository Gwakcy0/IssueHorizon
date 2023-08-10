import os
import pandas as pd
import warnings
from text_preprocessing import *

# def load_dataframes():
#     path = os.getcwd()
#     path = os.path.join(path, 'data')
#     print(f'PATH: {path}')

#     for dir in os.listdir(path):
#         p = os.path.join(path, dir)
#         if os.path.isdir(p):
#             print(f'========== Current dataset: {dir} ==========')
#             for file in os.listdir(p):
#                 with warnings.catch_warnings(record=True):
#                     warnings.simplefilter("always")
#                     yield dir, pd.read_excel(os.path.join(p, file), sheet_name=None, header=1, dtype={'작성일': str, '작성자': str, '제목': str, '내용': str, '댓글수': int})


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
#     df = clean_dataset(df)
#     df.to_csv(f'./data/clean_{dir}_{media}.csv', mode = 'w')


# 1. Make Clean Korean News article DataFrame
# 1-1. Lease 3 Law
# col_name = ['작성일', '작성자', '제목', '내용', '댓글수']
# data_path_1 = './data/임대차3법/임대차3법_2021년1월~2022년6월.xlsx'
# data_path_2 = './data/임대차3법/임대차3법_2020년7월~12월.xlsx'
# df_1 = pd.read_excel(data_path_1, sheet_name = 1, names = col_name, header = None)[2:]
# df_2 = pd.read_excel(data_path_2, sheet_name = 1, names = col_name, header = None)[2:]
# df = pd.concat([df_1, df_2], ignore_index = True)
# # Cleaned dataframe
# df = clean_dataset(df)
# print(df.head())
# # Save Cleaned DataFrame
# df.to_csv('./data/clean_news_1.csv', mode = 'w')



# # 2. Make Clean Korean News Corpus for pre-training
# # 2-1. Load all cleaned dataframe
# df_1 = pd.read_csv('./data/clean_news_1.csv')
# df_2 = pd.read_csv('./data/clean_news_2.csv')
# df_3 = pd.read_csv('./data/clean_news_3.csv')
# df_4 = pd.read_csv('./data/clean_news_4.csv')

# # 2-2. Extract Text from Text & Title columns
# df_1_text = df_1.Text.to_list()
# df_2_text = df_2.Text.to_list()
# df_3_text = df_3.Text.to_list()
# df_4_text = df_4.Text.to_list()

# total_news = df_1_text + df_1_text + df_1_text + df_1_text
# print("\n >> Korean News size : ", len(total_news))

# # 2-3. Make clean News Corpus for pretraining
# news_corpus = []
# for news in total_news:
#     news_sentences = news.split(".")
#     for sent in news_sentences:
#         news_corpus.append(sentence_split(sent))

# news_corpus = [x for x in news_corpus if x != 'None']

# print("\n >> Korean News Corpus size : ", len(news_corpus))

# # 2-4. List to a File line by line
# with open('./data/korean_news_corpus.txt', 'w') as fp:
#     for row in news_corpus:
#         fp.write("%s\n" % row)
#     print("Write is Done.")