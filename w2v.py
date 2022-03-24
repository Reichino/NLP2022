from nltk.corpus.reader import PlaintextCorpusReader
source_dir = "C_Chat/Sun Mar 20/"
pcr = PlaintextCorpusReader(root=source_dir, fileids=".*\.txt")

from nltk.probability import FreqDist
fd = FreqDist(samples=pcr.words())
print(fd.most_common(n=100))
print("/////////////////////////////")

source_dir2 = "Stock/Sun Mar 20/"
pcr = PlaintextCorpusReader(root=source_dir2, fileids=".*\.txt")

from nltk.probability import FreqDist
fd = FreqDist(samples=pcr.words())
print(fd.most_common(n=100))
# 1. What are the most common words in your two PTT boards respectively? 
#    Do they correspond to what you have expected?
#    (Please upload your Python script to GitHub, and paste your GitHub link on the online text of eCourse homework.)


from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
corpus = PathLineSentences(source_dir)


model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
print(model.wv.most_similar(positive=['推',"喜歡"], negative=['噓']))
print("/////////////////////////////")
corpus = PathLineSentences(source_dir2)
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)
print(model.wv.most_similar(positive=['台灣',"生息"], negative=['美國']))

# 2. By using the GenSim Word2Vec module, 
#    find a word x in an analogy like "man : king :: woman : x" (read: man is to king as woman is to x) 
#    in your PTT texts.
#    (Please upload your Python script to GitHub, and paste your GitHub link on the online text of eCourse homework.)
