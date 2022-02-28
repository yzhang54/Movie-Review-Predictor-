from sklearn.metrics.pairwise import cosine_similarity
import math
import pandas as pd

class KNN:

    k=0
    review_data=[]
    type_data=[]

    #the same as fit()
    def __init__(self, review_data,type_data):
        self.review_data= review_data
        self.type_data = type_data
        self.k =148
        #self.k= int(math.sqrt(review_data.shape[0])+1)

    #retrieve the number of nearest neighbors.
    def getNeighbors(self,tf_test_data):

        cos_result = cosine_similarity(tf_test_data,self.review_data)[0]
        #print(len(cos_result))

        dist_collection = []
        for index in range(len(cos_result)):
            dist_collection.append((self.type_data[index], cos_result[index]))
        
        dist_collection=sorted(dist_collection, key=lambda x:x[1],reverse=True)

        neighbors=[]

        for index in range(self.k):
            neighbors.append(dist_collection[index])
        return neighbors

    def getLabel(self, neighbors):
        pos=0
        neg=0
        for n in neighbors:
            if n[0]=="+1":
                pos+=1
            else:
                neg+=1
        
        if pos>=neg:
            return "+1"
        else:
            return "-1"
            

    def predict(self,tf_test_data):

        neighbors = self.getNeighbors(tf_test_data)

        return self.getLabel(neighbors)



        
        
            


    
         

