"""
Created on Wed Dec 25 17:53:39 2019

@author: Mohammad
"""
import numpy as np
import random
from sklearn.datasets import load_iris
import math

def cal_mean(data):
    mean = np.sum(data)/len(data)
    return mean

def cal_var(mean,data):
    variance = np.sum((data-mean)**2)/(len(data)-1)
    return variance
def gaussian(x, mean,var):
    return np.exp(-np.power(x - mean, 2.) / (2 * var)) / (2*math.pi*var)**0.5
def cal_mean_variance(data):
    mean_and_variance={}
    for j in range(3):
        mean_and_variance[j]=list()
        for i in range(4):
            d = data[j][:,i] 
            mean = cal_mean(d)
            variance = cal_var(mean,d)
            mean_and_variance[j].append([mean,variance])
    return mean_and_variance        
def seprate_train_test(index): 
    train_indicesfold=[]
    for i in which_fold[index][0:-1]: 
        train_indicesfold.extend(folds[i])
                
    test_index = which_fold[index][-1] 
    
    test_indicesfold = []
    test_indicesfold=folds[test_index]

    return test_indicesfold , train_indicesfold
def cal_precision_confusion_naive(indices):
    correct = 0
    conf = np.zeros((3,3))
    for p in indices:
        x_vector=data[p]
        aq=[]
        for i in range(3):
            product=1
            for j in range(4):
                product *=gaussian(x_vector[j],mean_and_variance[i][j][0],mean_and_variance[i][j][1])
            aq.append(product)
        
        predict = aq.index(max(aq))
        if(predict==targets[p]):
            correct+=1
            conf[targets[p],predict]+=1
        else:
            conf[targets[p],predict]+=1
            
    return correct/len(indices),conf
def cal_cov_mean(x): 
    cov_mat = np.zeros(shape=(4,4))
    x_bar =np.sum(x,axis=0)
    x_bar=np.reshape(x_bar,(4,1))/len(x) 
    cov_mat = np.cov(x,rowvar=0) 
    return [cov_mat,x_bar]
def gaus_bayes(x,class_number):
        cov_mat,x_bar = cov_means_class[class_number][0],cov_means_class[class_number][1]
        q = np.transpose(np.array([x])) - x_bar
        q_t = np.transpose(q)
        u = ((2*math.pi)**2) * ((np.linalg.det(cov_mat))**0.5)
        d = np.matmul(q_t ,np.linalg.inv(cov_mat))
        dd = -0.5*np.reshape(np.matmul(d,q),(1))
        return math.exp(dd)/u
def precision_confusion_bayes(indices):
    correct = 0
    conf=np.zeros((3,3))
    for p in indices:
        x_vector=data[p]
        aq=[]
        for i in range(3):
            w = gaus_bayes(x_vector,i)
            aq.append(w)
        predict = aq.index(max(aq))
        if(predict==targets[p]):
            correct+=1
            conf[targets[p],predict]+=1
        else:
            conf[targets[p],predict]+=1
            
    return correct/len(indices),conf
data,targets = load_iris(return_X_y=True)
class_indices= [[i for i in range(50)],[j for j in range(50,100)],[k for k in range(100,150)]]
which_fold=[[0,1,2,3,4],[0,1,2,4,3],[0,1,3,4,2],[0,2,3,4,1],[1,2,3,4,0]] 
folds={}
for i in range(5):
    b=[]
    for j in range(3):
        for k in range(10):
            index = random.choice(class_indices[j])
            b.append(index)
            class_indices[j].remove(index)
    folds[i]=b
indices_train70 = random.sample(range(0,50),math.ceil(0.7/3*len(data)))  
indices_train70.extend(random.sample(range(50,100),math.ceil(0.7/3*len(data))))
indices_train70.extend(random.sample(range(100,150),math.ceil(0.7/3*len(data))))
indices_test70=[]
for i in range(150):
    if(i not in indices_train70):
        indices_test70.append(i)
precisions_train = 0
precisions_test = 0
for i in range(5):
    test_indicesfold , train_indicesfold = seprate_train_test(i)
    indices_classfold={}
    for j in range(3):
        indices_classfold[j] = [k for k in train_indicesfold if(targets[k]==j)]
    classsfold = {0:data[indices_classfold[0]],1:data[indices_classfold[1]],2:data[indices_classfold[2]]}
    
    mean_and_variance = cal_mean_variance(classsfold)
    
    prec_train ,confusion_train = cal_precision_confusion_naive(train_indicesfold)
    print("training confusion matrix".format(i),confusion_train)
    print(prec_train)
    prec_test ,confusion_test = cal_precision_confusion_naive(test_indicesfold)
    print("test confusion matrix ".format(i),confusion_test)
    print(prec_test)
    
    precisions_train+=prec_train
    precisions_test+=prec_test

print("average precision in train " ,precisions_train/5)
print("average precision in test",precisions_test/5)
class70 = {0:data[indices_train70[0:35]],1:data[indices_train70[35:70]],2:data[indices_train70[70:105]]}
mean_and_variance = cal_mean_variance(class70)
prec_train70 , confusin_train70 = cal_precision_confusion_naive(indices_train70) 
prec_test70 , confusin_test70 = cal_precision_confusion_naive(indices_test70) 
print("70% data for training: ",prec_train70)
print(confusin_train70)
print("30% data for test: ",prec_test70)
print(confusin_test70)
precisions_train = 0
precisions_test = 0
for i in range(5):
    test_indicesfold , train_indicesfold = seprate_train_test(i)
    indices_classfold={}
    for j in range(3):
        indices_classfold[j] = [k for k in train_indicesfold if(targets[k]==j)] 
    classsfold = {0:data[indices_classfold[0]],1:data[indices_classfold[1]],2:data[indices_classfold[2]]} 
    
    cov_means_class={}
    for l in range(3):
        cov_means_class[l] = cal_cov_mean(classsfold[l])
    
    prec_train ,confusion_train = precision_confusion_bayes(train_indicesfold)
    print(" training confusion matrix ".format(i),confusion_train)
    print(prec_train)
    prec_test ,confusion_test = precision_confusion_bayes(test_indicesfold)
    print("test confusion matrix ".format(i),confusion_test)
    print(prec_test)
    
    precisions_train+=prec_train
    precisions_test+=prec_test
   
print("average precision in train " ,precisions_train/5)
print("average precision in test",precisions_test/5)
cov_means_class={}
for i in range(3):
    cov_means_class[i] = cal_cov_mean(class70[i])    
prec_train70 , confusin_train70 = precision_confusion_bayes(indices_train70) 
prec_test70 , confusin_test70 = precision_confusion_bayes(indices_test70) 
print("70% data for training: ",prec_train70)
print(confusin_train70)
print("30% data for test: ",prec_test70)
print(confusin_test70)