import codecs
import nltk
from nltk.collocations import *
from nltk.metrics.spearman import *

Gold_Standard = [('СУД', 'ПРИЗНАТЬ'), #77
                 ('СУД', 'УДОВЛЕТВОРИТЬ'), #51
                 ('СУД', 'ПРИГОВОРИТЬ'), #26
                 ('УДОВЛЕТВОРИТЬ', 'ИСК'), #22
                 ('ПРИНЯТЬ', 'РЕШЕНИЕ'), #20
                 ('ВЫНЕСТИ', 'РЕШЕНИЕ'), #15
                 ('НАЛОЖИТЬ', 'АРЕСТ'),  #11
                 ('СУД', 'ПОСТАНОВИТЬ'), #9
                 ('ОТКЛОНИТЬ', 'ИСК'), #7
                 ('ВЫНЕСТИ', 'ПРИГОВОР'), #6
                 ]
                 
Gold_RANKED = list(ranks_from_sequence(Gold_Standard))

                 
arr = []

f = codecs.open('court-V-N.csv', "r", "utf8")
for line in f:
    line = line.rstrip()
    words = line.split(",")
    doc = [x.replace(" ","") for x in words]
    arr.append(doc)
    
finder = BigramCollocationFinder.from_documents(arr)
finder.apply_freq_filter(3)
bigram_measures = nltk.collocations.BigramAssocMeasures()
bigrams = finder.nbest(bigram_measures.pmi, 10)

stopwords = nltk.corpus.stopwords.words('russian')
finder.apply_word_filter(lambda w: len(w) < 2 or w.lower() in stopwords)

print('---LOGLIKELIHOOD---')

LOGLIKELIHOOD = []

Log = finder.score_ngrams(bigram_measures.likelihood_ratio)
Log_vs_GS = []
for i in Log:
    for j in Gold_Standard:
        if i[0][0] == j[0] and i[0][1] == j[1]:
            Log_vs_GS.append(i)
            

for i in sorted(Log_vs_GS, key=lambda bi: bi[-1], reverse=True):
  LOGLIKELIHOOD.append(i[0])
  
for i in LOGLIKELIHOOD:
    print(i)
    
print('%0.1f' % spearman_correlation(ranks_from_sequence(Gold_Standard),
                                         ranks_from_sequence(LOGLIKELIHOOD)))

print('---STUDENT---')

STUDENT = []

Stud = finder.score_ngrams(bigram_measures.student_t)
Stud_vs_GS = []
for i in Stud:
    for j in Gold_Standard:
        if i[0][0] == j[0] and i[0][1] == j[1]:
            Stud_vs_GS.append(i)
            

for i in sorted(Stud_vs_GS, key=lambda bi: bi[-1], reverse=True):
  STUDENT.append(i[0])
  
for i in STUDENT:
    print(i)
    
print('%0.1f' % spearman_correlation(ranks_from_sequence(Gold_Standard),
                                         ranks_from_sequence(STUDENT)))

'''В итоге, обе метрики -  LogLikelihood и Распределение Стьюдента дали очень 
высокие ранговые коэффициенты корреляции - 0.8 и 0.9 соответственно. Если посмотреть
на ранжированные списки биграмм, полученные при помощи обеих метрик и сравнить их
с золотым стандартом, то можно заметить, что во всех случаях биграммы распределены
практически идентичным образом'''





