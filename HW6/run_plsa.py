# Repeat with dataset 2
# input file, outpot files and parameters
datasetFilePath = './plsa/dataset/dataset1.txt' # or set as './plsa/dataset/dataset2.txt'
stopwordsFilePath = './plsa/dataset/stopwords.dic'
docTopicDist = './plsa/output/docTopicDistribution.txt'
topicWordDist = './plsa/output/topicWordDistribution.txt'
dictionary = './plsa/output/dictionary.dic'
topicWords = './plsa/output/topics.txt'

K = 4   # number of topic
maxIteration = 20 # maxIteration and threshold control the train process
threshold = 3
topicWordsNum = 20 # parameter for output

# input file, outpot files and parameters
datasetFilePath = './plsa/dataset/dataset2.txt' # or set as './plsa/dataset/dataset2.txt'

from plsa.plsa import PLSA
from plsa.utils import preprocessing

N, M, word2id, id2word, X = preprocessing(datasetFilePath, stopwordsFilePath) # data processing

for K in range(40, 55):
    plsa_model = PLSA()
    plsa_model.initialize(N, K, M, word2id, id2word, X)

    oldLoglikelihood = 1
    newLoglikelihood = 1
    print(f"Iteration: ", end="", flush=True)
    for i in range(0, maxIteration):
        print(f"{i},", end="", flush=True)
        plsa_model.EStep() #implement E step
        plsa_model.MStep() #implement M step
        newLoglikelihood = plsa_model.LogLikelihood()
        if(abs(newLoglikelihood - oldLoglikelihood) < threshold):
            # change to absolute value, or else it terminates after first iteration
            break
        oldLoglikelihood = newLoglikelihood
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] "
          f"Log likelihood for dataset2, K={K}: {oldLoglikelihood}")

