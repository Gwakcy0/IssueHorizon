import re
import hanja
import pandas as pd

##############
# Clean Text #
##############
# 1. Filtering sentences based on words length
def sentence_split(sent):
    if len(sent.split()) > 5:
        return sent
    else:
        pass

# 2. Filtering Chinese characters
def cleansing_chinese(sent):
    # Chinese to Korean
    sent = hanja.translate(sent, 'substitution')
    return sent

# 3. Filtering special characters and spaces
def cleansing_special(sent):
    sent = re.sub("[],,ㆍ·\'\"’‘”“!?\\‘|\<\>`\'[\◇…@▶▲ⓒ]", " ", sent)
    sent = re.sub("[^.가-힣0-9\\s]", " ", sent)
    sent = re.sub("\s+", " ", sent)
    sent = sent.strip()
    sent = sent.replace("[SEP]", "")
    return sent

# 4. Final Preprocessing
def cleansing_sent(sent):
    clean_sent = cleansing_chinese(sent)
    clean_sent = cleansing_special(clean_sent)
    return clean_sent

# 5. Pre-processing
def preprocessing(text_list):
    # convert to string
    sent = list(map(str, text_list))
    # Filtering sentence length
    #sent = list(map(sentence_split, sent))
    # Filtering None
    sent = list(filter(None, sent))
    # Remove Chines and Special characters
    sent = list(map(cleansing_sent, sent))
    return sent

# 6. Examples
# sentence = '[1] 文대통령이 www abcde "실언"했다는 北김여정…아슬아슬한 (남북관계).'
# clean_sentence = cleansing_sent(sentence)

# print(">> Before Preprocessing : {}".format(sentence))
# print(">> After Preprocessing : {}".format(clean_sentence))


#######################################
# Clean Korean News article dataframe #
#######################################
def clean_dataset(dataframe):
    # 1. Drop NaN and duplicate data from Original dataset
    dataframe = dataframe.dropna(subset = ['내용'])
    dataframe = dataframe.drop_duplicates(subset = ['내용'])
    # dataframe = dataframe.drop_duplicates(subset = ['제목'])

    # 2. Datetime and order by Date
    dataframe['작성일'] = pd.to_datetime(dataframe['작성일'])
    dataframe = dataframe.sort_values(by = '작성일')

    # 3. text-Preprocessing
    text = dataframe['내용'].to_list()
    # title = dataframe['제목'].to_list()

    new_text = preprocessing(text)
    # new_title = preprocessing(title)

    # 4. Update Cleaned DataFrame
    dataframe['내용'] = new_text
    # dataframe['제목'] = new_title

    # 5. Drop NaN and duplicate data from Cleaned dataset
    dataframe = dataframe.dropna(subset = ['내용'])
    dataframe = dataframe.drop_duplicates(subset=['내용'])
    # dataframe = dataframe.drop_duplicates(subset=['제목'])
    print(">> Data Size :", len(dataframe))

    return dataframe