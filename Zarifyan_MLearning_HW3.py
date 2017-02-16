# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:51:19 2017

@author: Masha
"""
import numpy as np
import re
from matplotlib import pyplot as plt
from sklearn import grid_search, svm

def lenwords(sentence):
    return [len(word) for word in sentence.split()]


def lenletters(sentence): #1
    sentence = re.sub('[\.!,:;\?\-\'\"]', "", sentence)
    k = 0
    for word in sentence.split():
        k += len(word)
    return k

def different_letters(sentence): #2
    sentence = re.sub("[\.!,:;\?\-\'\"]", "", sentence)
    sentence = sentence.lower()
    k = 0
    letters = list(sentence)
    for i in set(letters):
        if i!=' ':
            k += 1
    return k

def vowels(sentence): #3
    sentence = re.sub("[\.!,:;\?\-\'\"]", "", sentence)
    k = 0
    sentence = sentence.lower()
    letters = list(sentence)
    for i in letters:
        if i!=' ':
            if i.lower() in 'аеёиоуыэюя':
                k += 1
    return k

def median_letters_word(s): #4
    s = s.lower()
    s = re.sub("[\.!,:;\?\-\'\"]", "", s)
    s = s.split()
    arr =[]
    for word in s:
        arr.append(len(word))
    return np.median(arr)

'''Эта функция считает гласные в предложении:''' 
def vowel_word(s):
    k = 0 
    letters = list(s)
    for letter in letters:
        if letter in 'аеёиоуыэюя':
             k+=1
    return k
'''А эта уже подсчитывает медиану гласных в предложении'''
def vowels_median(sent): #5
 arr = []
 sent = re.sub("[\.!,:;\?\-\'\"]", "", sent)
 sent = sent.lower()
 sent = sent.split()
 for s in sent:
   res = vowel_word(s)
   arr.append(res)
 return np.median(arr)

with open('anna.txt', encoding='utf-8') as f:
    anna = f.read()
with open('sonets.txt', encoding='utf-8') as f:
    sonets = f.read()

anna_sentences = re.split(r'(?:[.]\s*){3}|[.?!]', anna)
sonets_sentences = re.split(r'(?:[.]\s*){3}|[.?!]', sonets)


anna_data = [(lenletters(s), different_letters(s), vowels(s), median_letters_word(s), vowels_median(s)) for s in anna_sentences if len(lenwords(s)) > 0]
sonets_data = [(lenletters(s), different_letters(s), vowels(s), median_letters_word(s), vowels_median(s)) for s in sonets_sentences if len(lenwords(s)) > 0]
#print(anna_data[:10])
#print(sonets_data[:10])

anna_data = np.array(anna_data)
sonets_data = np.array(sonets_data)
'''тут пример одного из графиков. Сравниваем длину предложения в буквах и медиану числа букв в слове '''
#plt.figure()
#plt.plot(anna_data[:,0], anna_data[:,3], 'og', sonets_data[:,0], sonets_data[:,3], 'sb')
#plt.show() 
'''Как мы видим, тексты не сильно различаются. Единственное, что бросается в глаза при 
построении любых графиков - это то, что в Анне Карениной в разы больше слов, чем в сонетах Шекспира, и это 
немного мешает нам определить, насколько [не]похожи друг на друга эти два корпуса.
'''
'''Почему-то ничего не работает на полных текстах, поэтому беру одинаковые части из каждого корпуса'''
anna_data1 = np.array(anna_data[:30])
sonets_data1 = np.array(sonets_data[:30])
data = np.vstack((anna_data1, sonets_data1))
#parameters = {'C': (0.65, 0.75, 0.8)}
parameters = {'C': (0.3, 0.3, 0.3, 0.4)}
gs = grid_search.GridSearchCV(svm.LinearSVC(), parameters)
gs.fit(data[:, 1:], data[:, 0])
print('Best result is ',gs.best_score_)
print('Best C is', gs.best_estimator_.C)

clf = svm.LinearSVC(C=gs.best_estimator_.C)
clf.fit(data[::2, 1:], data[::2, 0])

#best C is 0.4
wrong = 0
for obj in data[1::2, :]:
    label = clf.predict(obj[1:])
    if label != obj[0] and wrong < 3:
        print('Пример ошибки машины: class = ', obj[0], ', label = ', label, ', экземпляр ', obj[1:])
        wrong += 1
    if wrong > 3:
       break

'''
Пример ошибки машины: class =  26.0 , label =  [ 121.] , экземпляр  [ 16.   9.   4.   2.]
Пример ошибки машины: class =  107.0 , label =  [ 165.] , экземпляр  [ 21.   47.    5.5   2. ]
Пример ошибки машины: class =  51.0 , label =  [ 121.] , экземпляр  [ 21.  22.   4.   2.]
'''