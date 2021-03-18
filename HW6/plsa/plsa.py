from numpy import zeros, int8, log
import numpy as np
from pylab import random
import sys
#import jieba
import nltk
from nltk.tokenize import word_tokenize 
import re
import time
import codecs

class PLSA(object):
    def initialize(self, N, K, M, word2id, id2word, X):
        self.word2id, self.id2word, self.X = word2id, id2word, X
        self.N, self.K, self.M = N, K, M
        # theta[i, j] : p(zj|di): 2-D matrix
        self.theta = random([N, K])
        # beta[i, j] : p(wj|zi): 2-D matrix
        self.beta = random([K, M])
        # p[i, j, k] : p(zk|di,wj): 3-D tensor
        self.p = zeros([N, M, K])
        for i in range(0, N):
            normalization = sum(self.theta[i, :])
            for j in range(0, K):
                self.theta[i, j] /= normalization

        for i in range(0, K):
            normalization = sum(self.beta[i, :])
            for j in range(0, M):
                self.beta[i, j] /= normalization


    def EStep(self):
        for i in range(0, self.N): # w
            for j in range(0, self.M): # d
                ## ================== YOUR CODE HERE ==========================
                ###  for each word in each document, calculate its
                ###  conditional probability belonging to each topic (update p)
                # total = 0
                # for k in range(0, self.K):
                #     total += self.theta[i, k] * self.beta[k, j]
                
                # for k in range(self.K):
                #     self.p[i, j, k] = self.theta[i, k] * self.beta[k, j] / total

                # vectorized version
                total = np.matmul(self.theta[i], self.beta[:,j].T)
                self.p[i, j, :] = self.theta[i] * self.beta[:, j] / total
                # ============================================================

    def MStep(self):
        # update beta
        for k in range(0, self.K):
            # ================== YOUR CODE HERE ==========================
            ###  Implement M step 1: given the conditional distribution
            ###  find the parameters that can maximize the expected likelihood (update beta)
            # denominator = 0
            # for j in range(self.M):
            #     # for i in range(self.N):
            #     #     denominator += self.p[i, j, k] * self.X[i, j]
            #     denominator += np.matmul(self.p[:, j, k], self.X[:, j])
            
            # for j in range(self.M):                    
            #     # numerator = 0
            #     # for i in range(self.N):
            #     #     numerator += self.p[i, j, k] * self.X[i, j]
            #     print(self.p[:,j,k].shape)
            #     print(self.X[:,j].shape)
            #     numerator = np.matmul(self.p[:, j, k], self.X[:, j])
            #     self.beta[k, j] = numerator / denominator

            # vectorized version
            denominator = np.einsum('ij,ij', self.p[:, :, k], self.X)
            self.beta[k] = np.einsum('i...,i...', self.p[:, :, k], self.X) / denominator
            # ============================================================
        
        # update theta
        for i in range(0, self.N):
            # ================== YOUR CODE HERE ==========================
            ###  Implement M step 2: given the conditional distribution
            ###  find the parameters that can maximize the expected likelihood (update theta)
            # for k in range(self.K):
            #     # numerator = 0
            #     # for j in range(self.M):
            #     #     numerator += self.p[i, j, k] * self.X[i, j]
            #     numerator = np.matmul(self.p[i, :, k], self.X[i, :].T)
            #     self.theta[i, k] = numerator / self.N
            
            # denominator = 0
            # for k in range(0, self.K):
            #     for j in range(0, self.M):
            #         denominator += self.p[i, j, k] * self.X[i, j]

            # vectorized version
            denominator = self.X[i,:].sum()
            numerator = np.matmul(self.p[i].T, self.X[i])
            self.theta[i] = numerator / denominator       
            # ============================================================

    # calculate the log likelihood
    def LogLikelihood(self):
        loglikelihood = 0
        # for i in range(0, self.N):
        #     # for j in range(0, self.M):
        #         # # ================== YOUR CODE HERE ==========================
        #         # ###  Calculate likelihood function
        #         # # word_prob = 0
        #         # # for k in range(0, self.K):
        #         # #     word_prob += self.theta[i, k] * self.beta[k, j]
        #         # word_prob = np.matmul(self.theta[i], self.beta[:,j].T)
        #         # loglikelihood += self.X[i, j] * np.log(word_prob)
        #         # # ============================================================
        #     product = np.matmul(self.beta.T, self.theta[i])
        #     # loglikelihood += np.multiply(self.X[i], np.log(product)).sum()
        #     loglikelihood += np.einsum('i,i', self.X[i], np.log(product))

        # vectorized version
        product = np.matmul(self.theta, self.beta)
        loglikelihood += np.einsum('ij,ij', self.X, np.log(product))
        return loglikelihood

    # output the params of model and top words of topics to files
    def output(self, docTopicDist, topicWordDist, dictionary, topicWords, topicWordsNum):
        # document-topic distribution
        file = codecs.open(docTopicDist,'w','utf-8')
        for i in range(0, self.N):
            tmp = ''
            for j in range(0, self.K):
                tmp += str(self.theta[i, j]) + ' '
            file.write(tmp + '\n')
        file.close()
        
        # topic-word distribution
        file = codecs.open(topicWordDist,'w','utf-8')
        for i in range(0, self.K):
            tmp = ''
            for j in range(0, self.M):
                tmp += str(self.beta[i, j]) + ' '
            file.write(tmp + '\n')
        file.close()
        
        # dictionary
        file = codecs.open(dictionary,'w','utf-8')
        for i in range(0, self.M):
            file.write(self.id2word[i] + '\n')
        file.close()
        
        # top words of each topic
        file = codecs.open(topicWords,'w','utf-8')
        for i in range(0, self.K):
            topicword = []
            ids = self.beta[i, :].argsort()
            for j in ids:
                topicword.insert(0, self.id2word[j])
            tmp = ''
            for word in topicword[0:min(topicWordsNum, len(topicword))]:
                tmp += word + ' '
            file.write(tmp + '\n')
        file.close()