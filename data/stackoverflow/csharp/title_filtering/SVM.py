#===============================================================================
# Based on
# Author : Rishi Josan
# CSE 628: Natural Language Processing
# Stony Brook University
# March 8, 2014
# original version at https://github.com/rishijosan/sentimentAnalysis
#===============================================================================

import nltk
from  nltk.probability import FreqDist
from collections import OrderedDict
import numpy as np
from sklearn import svm
import re
from nltk import PorterStemmer
porter = PorterStemmer()
from nltk.corpus import stopwords
import sys


class SVM:

    @staticmethod
    def tokenize(text):
      # text = NB.remove_punctuation(text)
      try:
        text = text.decode('utf-8').encode('ascii', 'replace').strip().lower()
      except:
        text = text.encode('ascii', 'replace').strip().lower()
      word = [porter.stem(w) for w in re.findall(r"[\w'-]+|[^\s\w]", text)]   # split punctuations but dont split single quotes for words like don't
      biword =  [b for b in nltk.bigrams(word)]
      triword =  [t for t in nltk.trigrams(word)]
      # word = [w for w in word if w not in stopwords.words('english')]
      return  word # triword


    def train(self, posTrainCorpus, negTrainCorpus):
        tokens = []

        fp = open(posTrainCorpus, 'r')
        for line in fp:
          tokens += SVM.tokenize(line)
        fp.close()

        fn = open(negTrainCorpus, 'r')
        for line in fn:
          tokens += SVM.tokenize(line)
        fn.close()

        #Create Frequency Distribution from both Positive and Negative Corpora
        trainFreq = nltk.FreqDist(tokens) 

        #No of Features
        self.noFeat = len(trainFreq)

        #Get Keys to maintain Order
        self.trainKeys = trainFreq.keys()

        #Create OrderedDict for features: Use this as sample for all files
        ordFeat = OrderedDict()
        for key in trainFreq.keys():
            ordFeat.update( {key: trainFreq.freq(key)} )

        posFeatList = self.featureList(posTrainCorpus)
        negFeatList = self.featureList(negTrainCorpus)
        featList = posFeatList + negFeatList

        noPos = len(posFeatList)
        noNeg = len(negFeatList)

        labels = []

        for j in range(noPos):
            labels.append(1)
        for k in range(noNeg):
            labels.append(0)

        #Create numpy Array for word frequencies : Feature Vector
        trainFreqArr = np.array(featList)
        trainLabels = np.array(labels)


        #Fit SVM
        # docClassifier = svm.SVC( C=1000)
        self.docClassifier = svm.LinearSVC()
        self.docClassifier.fit(trainFreqArr, trainLabels) 


    def getFeat(self, line):
        listItem = [0]*self.noFeat
        fileFreqDist = nltk.FreqDist(SVM.tokenize(line))

        i = 0
        for key in self.trainKeys:
            if fileFreqDist.has_key(key):
                listItem[i] = fileFreqDist.get(key)
            i = i + 1
        return listItem

    def featureList(self, corpus):
        featList = []
        f = open(corpus, 'r')
        for line in f:
            featList.append(self.getFeat(line))
        f.close()

        return featList


    def test(self, posTestCorpus, negTestCorpus):
        posTestFeatList = self.featureList(posTestCorpus)
        negTestFeatList = self.featureList(negTestCorpus)

        posTestarr = np.array(posTestFeatList)
        negTestarr = np.array(negTestFeatList)

        # prediction result stored in array which is the converted to list and added to opt list
        print "Bad identify rate = " + str(sum(np.array(self.docClassifier.predict(posTestarr)).tolist()) / 116.0)
        print "Good elimination rate = " + str(sum(np.array(self.docClassifier.predict(negTestarr)).tolist()) / 116.0)
        print "Accuracy  = " + str((sum(np.array(self.docClassifier.predict(posTestarr)).tolist()) + 116 - sum(np.array(self.docClassifier.predict(negTestarr)).tolist() )) / 232.0)

    def filter(self, sent):
        testFeatList = []
        testFeatList.append(self.getFeat(sent))
        testarr = np.array(testFeatList)
        opt = np.array(self.docClassifier.predict(testarr)).tolist()
        return opt[0]


if __name__ == '__main__':
    s = SVM()
    s.train("balanced/pos_train.txt", "balanced/neg_train.txt")
    s.test("balanced/pos_test.txt", "balanced/neg_test.txt")
    print s.filter("How can i print an array?")
    print s.filter("My code does not run")
