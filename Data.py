
from __future__ import division
from operator import truediv
import numpy as np
import string
import nltk
nltk.download('stopwords')
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
CLEANR = re.compile('<.*?>') 
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from KNN import KNN
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import statistics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score
import math


class Data:
        
        file_data =[]
        test_file_data=[]

        def __init__(self):
            self.load_file()


        def load_file(self):
            #extract all info from the file.
            file = open("train_file.txt",encoding="utf8") 
            file_data = file.read()
            file.close()
            self.file_data = np.array(file_data.split('\n'))


            file = open("test_file.txt",encoding="utf8")
            test_file_data = file.read()
            file.close()
            self.test_file_data = np.array(test_file_data.split('\n'))
            
            


        def stemming(self,review):
            #stemming
            ps = PorterStemmer()
            words_dict = set()
            for w in review:
                stem = ps.stem(w)
                words_dict.add(stem)

                
            return words_dict

        def remove_html(self,review):
            return re.sub(CLEANR, '', review)


        def remove_punc_num(self,review):
            tokens = wordpunct_tokenize(review)
            tokens = [word for word in tokens if word.isalpha()]
            #tokens = [w for w in tokens if not w in string.punctuation]
            return tokens

        def remove_single_char(self, words_dict):
        
            new_text = ""
            for w in words_dict:
                if len(w) > 1:
                    new_text = new_text + " " + w
                    print(new_text)

            return new_text

        def data_cleaning(self,review_data):
            result= []
            for index in range(len(review_data)):

                review = review_data[index]
                review = self.remove_html(review)
                words_dict = self.remove_punc_num(review)
                words_dict = self.stemming(words_dict)
                
                result.append(" ".join(words_dict))

            return result


        def single_performance(self,actual, predicted):

            accuracy = accuracy_score(actual, predicted)
            precision,recall,fscore,support=score(actual,predicted,average='macro')

            return accuracy,fscore

                

            

if __name__ == "__main__":

    
    """
    dataobj = Data()
    knnobj=0
    tf_idf=0
    test_review_data=[]
    tf_train_review_data =[]
    accuracy_list=[]
    fscore_list=[]

    #cross validation
    kf = KFold(n_splits=10, shuffle = True,random_state=43)
    for train,test in kf.split(dataobj.file_data):

        
        #train data: split label and review 
        train_data = np.char.split(dataobj.file_data[train],sep='\t')
        train_type_data = [row[0] for row in train_data]
        train_review_data = [row[1] for row in train_data]
        train_review_data = dataobj.data_cleaning(train_review_data)

        tf_idf = TfidfVectorizer(stop_words='english',analyzer='word',use_idf=True)
        tf_train_review_data = tf_idf.fit_transform(train_review_data)
   

        #test data: split label and review
        test_data = np.char.split(dataobj.file_data[test],sep='\t')
        test_type_data = [row[0] for row in test_data]
        test_review_data = [row[1] for row in test_data]
        test_review_data= dataobj.data_cleaning(test_review_data)


        knnobj = KNN(tf_train_review_data, train_type_data)
        classification_result =[]


        for index in range(len(test_review_data)):
            tf_test_review_data = tf_idf.transform([test_review_data[index]])
            classification_result.append(knnobj.predict(tf_test_review_data))


        accuracy,fscore = dataobj.single_performance(test_type_data, classification_result)
        accuracy_list.append(accuracy)
        fscore_list.append(fscore)
        

    accuracy_avg = statistics.mean(accuracy_list)
    fscore_avg= statistics.mean(fscore_list)

    print('avg of accuracy: ',accuracy_avg)
    print('avg of fscore: ',fscore_avg)


    #testing different k value 
    k = int(math.sqrt(tf_train_review_data.shape[0])+1)
    k_list=[]
    accuracy_list=[]
    fscore_list=[]
    for index in range(k-100,k+200,10):
 
        classification_result =[]
        knnobj.k = index
        k_list.append(index)


        for index in range(len(test_review_data)):
            tf_test_review_data = tf_idf.transform([test_review_data[index]])
            classification_result.append(knnobj.predict(tf_test_review_data))

        accuracy,fscore = dataobj.single_performance(test_type_data, classification_result)
        accuracy_list.append(accuracy*100)
        fscore_list.append(fscore*100)


    plt.plot(accuracy_list,k_list,label='accuracy', linewidth=5)
    plt.plot(fscore_list,k_list,label='fscore', linewidth=5)
    plt.ylabel('K value')
    plt.legend(loc="upper left")
    plt.show()

    """
    #Comment out this whole section if you want to try predicting for the test data file. And comment out the above lines of code.
    dataobj = Data()
    train_data = np.char.split(dataobj.file_data,sep='\t')
    train_type_data = [row[0] for row in train_data]
    train_review_data = [row[1] for row in train_data]
    train_review_data = dataobj.data_cleaning(train_review_data)

    tf_idf = TfidfVectorizer(stop_words='english',analyzer='word',use_idf=True)
    tf_train_review_data = tf_idf.fit_transform(train_review_data)
    knnobj = KNN(tf_train_review_data, train_type_data)

    pos =0
    neg =0
    f = open("result.txt", "w", encoding='utf8')

    test_data = dataobj.data_cleaning(dataobj.test_file_data)

    for index in range(len(test_data)):

        tf_test_review_data = tf_idf.transform([test_data[index]])
        result = knnobj.predict(tf_test_review_data)

        if result == "+1":
            pos+=1
        else:
            neg+=1
        f.write(result+"\n")
    
    f.close()



