import sys
import os

sys.path.append(os.getcwd())

import tensorflow as tf
from tensorflow.keras import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from utilss.Preprocess import Preprocess

# 학습 파일 불러오기
def read_file(file_name):
    sents = []
    with open(file_name, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        for idx, l in enumerate(lines):
            if l[0] == ';'and lines[idx + 1][0] == '$':
                this_sent = []
            elif l[0] == '$' and lines[idx - 1][0] == ';':
                continue
            elif l[0] == '\n':
                sents.append(this_sent)
            else:
                this_sent.append(tuple(l.split()))
    return sents

# 전처리 객체 생성
p = Preprocess(word2index_dic='train_tools/dict/chatbot_dict.bin', userdic='utilss/user_dic.tsv')

# 학습용 말뭉치 데이터를 불러옴
corpus = read_file('models/ner/ner_train.txt')

# 말뭉치 데이터에서 단어와 BIO 태그만 불러와 학습용 데이터셋 생성
sentences, tags = [], []
for t in corpus:
    tagged_sentence = []
    sentence, bio_tag = [], []
    for w in t:
        tagged_sentence.append((w[1], w[3]))
        sentence.append(w[1])
        bio_tag.append(w[3])

    sentences.append(sentence)
    tags.append(bio_tag)

# 확인
print('샘플 크기 : \n', len(sentences))
print('0번째 샘플 문장 시퀀스 : \n', sentences[0])
print('0번째 샘플 bio 태그 : \n', tags[0])
print('샘플 문장 시퀀스 최대 길이 :', max(len(l) for l in sentences))
print('샘플 문장 시퀀스 평균 길이 :', (sum(map(len, sentences))/len(sentences)))

# 토크나이저 정의
tag_tokenizer = preprocessing.text.Tokenizer(lower=False)
# 소문자로 변환 X
tag_tokenizer.fit_on_texts(tags)

# 단어 사전 및 태그 사전 크기
vocab_size = len(p.word_index) + 1
tag_size = len(tag_tokenizer.word_index) + 1
print("BIO 태그 사전 크기 :", tag_size)
print("단어 사전 크기 :", vocab_size)

# 학습용 단어 시퀀스 생성
x_train = [p.get_wordidx_sequence(sent) for sent in sentences]
y_train = tag_tokenizer.texts_to_sequences(tags)

index_to_ner = tag_tokenizer.index_word
index_to_ner[0] = 'PAD'
# 시퀀스 패딩 처리
max_len = 40
x_train = preprocessing.sequence.pad_sequences(x_train, padding = 'post', maxlen=max_len)
y_train = preprocessing.sequence.pad_sequences(y_train, padding = 'post', maxlen=max_len)

# 학습 데이터와 테스트 데이터를 8 : 2 비율
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, random_state=1234)

# 출력 데이터를 원-핫 인코딩
y_train = tf.keras.utils.to_categorical(y_train, num_classes=tag_size)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=tag_size)

print('학습 샘플 시퀀스 형상: ', x_train.shape)
print('학습 샘플 레이블 형상: ', y_train.shape)
print('테스트 샘플 시퀀스 형상: ', x_test.shape)
print('테스트 샘플 레이블 형상: ', y_test.shape)

# 모델 정의(Bi-LSTM)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=30, input_length=max_len, mask_zero=True))
model.add(Bidirectional(LSTM(200, return_sequences=True, dropout=0.5, recurrent_dropout=0.25)))
model.add(TimeDistributed(Dense(tag_size, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=10)

print("평가 결과: ", model.evaluate(x_test, y_test)[1])
model.save('models/ner/ner_model.h5')

# 시퀀스를 NER 태그로 변환
def sequences_to_tag(sequences):
  result = []
  for sequence in sequences:
    temp = []
    for pred in sequence:
      pred_index = np.argmax(pred)
      temp.append(index_to_ner[pred_index].replace("PAD", "O"))
    result.append(temp)
  return result

from seqeval.metrics import f1_score, classification_report
# 테스트 데이터셋의 NER 예측
y_predicted = model.predict(x_test)
pred_tags = sequences_to_tag(y_predicted)
test_tages = sequences_to_tag(y_test)

print(classification_report(test_tags, pred_tags))
print("F1-score: {:.1%}".format(f1_score(test_tags, pred_tags)))