import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NB_model():
    def __init__(self): 
        self.pi = {} # to store prior probability of each class 
        self.Pr_dict = None
        self.num_vocab = None
        self.num_classes = None
    
    def fit(self, train_data, train_label, vocab, if_use_smooth=True):
        # get prior probabilities
        self.num_vocab = len(vocab['index'].tolist())
        self.get_prior_prob(train_label)
        # ================== YOUR CODE HERE ==========================
        # Calculate probability of each word based on class 
        # Hint: Store each probability value in matrix or dict: self.Pr_dict[classID][wordID] or Pr_dict[wordID][classID])
        # Remember that there are possible NaN or 0 in Pr_dict matrix/dict. Use smooth method
        self.Pr_dict = {}
        self.zero_word_dict = {}
        classes = set(train_data['classIdx'])
        print(f"Training class ", end="", flush=True)
        for c in classes:
            df_class = train_data.loc[train_data['classIdx'] == c]
            print(f"{c} ", end="", flush=True)
            total = df_class['count'].sum()
            self.Pr_dict[c] = df_class.groupby('wordIdx')['count'].sum().to_dict()
            self.zero_word_dict[c] = 1 / (total + self.num_vocab)
            for k in self.Pr_dict[c]:
                self.Pr_dict[c][k] = (self.Pr_dict[c][k] + 1) / (total + self.num_vocab)
        # ============================================================
        print("\nTraining completed!")
    
    def predict(self, test_data):
        test_dict = test_data.to_dict() # change dataframe to dict
        new_dict = {}
        prediction = []
        
        for idx in range(len(test_dict['docIdx'])):
            docIdx = test_dict['docIdx'][idx]
            wordIdx = test_dict['wordIdx'][idx]
            count = test_dict['count'][idx]
            try: 
                new_dict[docIdx][wordIdx] = count 
            except:
                new_dict[test_dict['docIdx'][idx]] = {}
                new_dict[docIdx][wordIdx] = count
                ''
        for docIdx in range(1, len(new_dict)+1):
            if docIdx % 2500 == 0:
                print(f"Finished predictions for doc {docIdx}")
            score_dict = {}
            #Creating a probability row for each class
            for classIdx in range(1,self.num_classes+1):
                score_dict[classIdx] = 0
                # ================== YOUR CODE HERE ==========================
                ### Implement the score_dict for all classes for each document
                ### Remember to use log addtion rather than probability multiplication
                ### Remember to add prior probability, i.e. self.pi
                score_dict[classIdx] = np.log(self.pi[classIdx])
                for w in new_dict[docIdx]:
                    score_dict[classIdx] += np.log(self.Pr_dict[classIdx].get(w, self.zero_word_dict[classIdx]))
                # ============================================================
            max_score = max(score_dict, key=score_dict.get)
            prediction.append(max_score)
        return prediction
                    
    
    def get_prior_prob(self,train_label, verbose=True):
        # print(train_label)
        unique_class = list(set(train_label))
        self.num_classes = len(unique_class)
        total = len(train_label)
        for c in unique_class:
            # ================== YOUR CODE HERE ==========================
            ### calculate prior probability of each class ####
            ### Hint: store prior probability of each class in self.pi
            self.pi[c] = 0
            for label in train_label:
                if c == label:
                    self.pi[c] += 1
            self.pi[c] = self.pi[c] / total
            # ============================================================
        if verbose:
            print("Prior Probability of each class:")
            print("\n".join("{}: {}".format(k, v) for k, v in self.pi.items()))
