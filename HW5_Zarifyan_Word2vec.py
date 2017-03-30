# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 18:06:18 2017

@author: Masha
"""
import numpy as np 
import pandas as pd
import os
import gensim
import logging 
import codecs
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

anecdotes = []
izvestia = []
tehmol = []

#m = 'ruwikiruscorpora_0_300_20.bin.gz'
#model = gensim.models.KeyedVectors.load_word2vec_format(m, binary=True)

#def vectorization(f):
 #   vect = []
   # for line in f:
     #line = line.split()
     #for w in line:
      #  if w in model:
       #     vect.append(model[w])
   # mean = sum(vect) / len(vect)
    #return mean

'''
Я лемматизировала и токенизировала тексты с помощью MyStem, прогоняя его по папкам, поэтому тут уже открываю файлы с 
обработанными текстами:
    Проходим по папкам и создаем датасет из текстов и категории, которой принадлежит текст.
'''
path_anecdots = r'D:/HSE_2016-2017/NLP_2017/HW5/CLEAN_DATA/aanekdotes_tokenized/'
path_izvestia = r'D:/HSE_2016-2017/NLP_2017/HW5/CLEAN_DATA/izvest_tokenized/'
path_tehmol = r'D:/HSE_2016-2017/NLP_2017/HW5/CLEAN_DATA/tehmol_tokenized/'
path = r'D:/HSE_2016-2017/NLP_2017/HW5/CLEAN_DATA/'


df = []

for folder in os.listdir(path):
    for name in os.listdir(path + folder):
        #with open(path + folder + '/' + name, 'r') as f:
        #try:
            f = codecs.open(path + folder + '/' + name, "r", "utf-8")
            df.append([folder, f.read()])
        #except:
            #print(folder, name)

df = pd.DataFrame(df, columns=['folder',  'text'])
print(df.head())

m = 'ruscorpora_1_300_10.bin.gz'



stories = df['text']
all_data = '\n'.join(stories)
'''
MyStem лемматизирует слова, обрамляя их в скобки. Очистим данные от скобок:
'''
all_data = all_data.replace('{', ' ')
all_data = all_data.replace('}', ' ')
all_data = all_data.replace('|', ' ')
w = codecs.open('big_output.txt', 'w', 'utf-8') 
w.write(all_data)

all_data = gensim.models.doc2vec.TaggedLineDocument('big_output.txt')
model = gensim.models.doc2vec.Doc2Vec(all_data, size=300, min_count=10, iter=50)
model.save('mymodel')
model = gensim.models.Doc2Vec.load('mymodel')

Vector_list = [v for v in model.docvecs]


X_train, X_test, y_train, y_test = train_test_split(Vector_list, df['class'], test_size=0.3)
clf = LogisticRegression(penalty="l2", solver="lbfgs", multi_class="multinomial", max_iter=300, n_jobs=4)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print(classification_report(y_test, predicted))
