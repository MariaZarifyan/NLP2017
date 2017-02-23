# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 20:27:56 2017

@author: Masha
"""

import nltk 
from nltk.corpus import wordnet
from nltk.wsd import lesk

print("1.Найти все значения (синсеты) для лексемы plant")
sets = wordnet.synsets('plant')
#print(sets)

print('2.Найти определение для лексемы plant в значении (а) "завод" и в значении (b) "растение"')
set1 = wordnet.synset('plant.n.01')
zavod = set1.definition()
#print(zavod) #завод

set2 = wordnet.synset('plant.n.02')
plant = set2.definition()
#print(plant) #растение

print('3. ..два произвольных контекста для (a)"завод" и (b)"растение"...алгоритмa Леска для разрешения неоднозначности')
plant1 = "I water my plants, flowers, trees and other living organisms every day".split()
zavod2 = "Two more car-assembly plants and industrial factories were closed by the strike".split()
print (lesk(plant1, 'plant', 'n').definition())
print (lesk(zavod2, 'plant', 'n').definition())
'''
Алгоритм работает плохо. Сначала он не угадал правильно ни одного значения, а потом
я решила поэкспериментировать и добавила в первое предложение часть определения лексемы,
в значении 'растение': 'living organisms',  и он сразу угадал значение правильно.
Вывод: Возможно, при выборе значения лексемы алгоритм Леска ориентируется не на семантику слов
находящихся рядом с лексемой, а только на слова из его определения в Wordnet-e, и поэтому так 
плохо работает.
'''
print('4. Найдите гиперонимы для значения (a) и гиперонимы для значения (b)')
print (set1.hypernyms())
print (set2.hypernyms())

print("5.")
#Найти min (d(plant: "завод", industry), d(plant: "завод", leaf)),
#а также min (d(plant: "растение", industry), d(plant: "растение", leaf))
ind = wordnet.synsets('industry')
for i in ind:
 print(i, i.definition())
 print(set1.path_similarity(i))
 print(set2.path_similarity(i))
print('OТВЕТ: min(d(plant: "завод", industry) 0.0714 - между plant.n.01 и лексемой diligence.n.02') 
print('OТВЕТ: min(d(plant: "растение", industry) 0.769 - между plant.n.02 и лексемой diligence.n.02') 
leaf = wordnet.synsets('leaf')
for i in leaf:
    print(i, i.definition())
    print(set1.path_similarity(i))
    print(set2.path_similarity(i))
print('OТВЕТ:min(d(plant: "завод", leaf)) - 0.111 - между plant.n.01 и лексемой leaf.n.02')
print('OТВЕТ:min(d(plant: "растение", leaf)) - 0.125 - между plant.n.02 и лексемой leaf.n.01 и leaf.n.03')

print('6.Вычислить двумя разными способами расстояние...')
master = wordnet.synsets("rattlesnake's master")
#print(master) - Выдает пустой массив, =>  синсета rattlesnake's master в Ворднете нет
org  = wordnet.synsets("organism")
whole = wordnet.synsets("whole")
#print(org)
#print(whole)   
org1 = wordnet.synset ('organism.n.01')#a living thing that has (or can develop) the ability to act or function independently
org2 = wordnet.synset ('organism.n.02')#a system considered analogous in structure or function to a living body
whole1 = wordnet.synset ('whole.n.01')#all of something including all its component elements or parts
whole2 = wordnet.synset ('whole.n.02')#an assemblage of parts that is regarded as a single entity
print('path-similarity')
print (org1.path_similarity(whole1))
print (org1.path_similarity(whole2))
print (org2.path_similarity(whole1))
print (org2.path_similarity(whole2))
print('shortest_path_distance')
print (org1.shortest_path_distance(whole1))
print (org1.shortest_path_distance(whole2))
print (org2.shortest_path_distance(whole1))
print (org2.shortest_path_distance(whole2))
print('Leacock-Chodorow Similarity')
print (org1.lch_similarity(whole1))
print (org1.lch_similarity(whole2))
print (org2.lch_similarity(whole1))
print (org2.lch_similarity(whole2))
'''
ВЫВОД: Мы сравнили расстояния между whole и organism с помощью трёх различных метрик: 
path-similarity, shortest_path_distance и Leacock-Chodorow Similarity. Посмотрев на полученные цифры, мы 
отчётливо видим, что все три метрики показывают наибольшую семантическую близость между лексемой organism.n.01
a living thing that has (or can develop) the ability to act or function independently
и лексемой whole.n.02
an assemblage of parts that is regarded as a single entity
Как мне кажется, данные результаты правильно отображают интуитивное представление о семантчиеской близости данных лексем,
так как глядя на определение лексемы whole.n.02 'целостность частей, воспринимаемая как единое существо', первая ассоциация,
которая приходит в голову - это организм, целостное живое существо.
'''