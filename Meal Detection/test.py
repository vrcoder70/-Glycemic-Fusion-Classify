import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statistics import mean
import math
from scipy.fftpack import fft,ifft,rfft
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, RepeatedKFold
from joblib import dump, load


def feature_matrix(test):
    index=test.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>5).dropna().index
    
    data = test.drop(test.index[index]).reset_index().drop(columns='index')
    
    index = data.isna().sum(axis=1).replace(0,np.nan).dropna().index
    
    data = data.drop(data.index[index]).reset_index().drop(columns='index')
    
    powerFirstMax=[]
    indexFirstMax=[]
    powerSecondMax=[]
    indexSecondMax=[]
    powerThirdMax=[]

    for i in range(len(data)):
        array=abs(rfft(data.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        
        sortedArray=abs(rfft(data.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sortedArray.sort()
        
        powerFirstMax.append(sortedArray[-2])
        powerSecondMax.append(sortedArray[-3])
        powerThirdMax.append(sortedArray[-4])
        
        indexFirstMax.append(array.index(sortedArray[-2]))
        indexSecondMax.append(array.index(sortedArray[-3]))
    

    featureMatrix = pd.DataFrame()
    featureMatrix['power_second_max']=powerSecondMax
    featureMatrix['power_third_max']=powerThirdMax
    
    
    secondDifferentialData=[]
    standardDeviation=[]
    for i in range(len(data)):
        secondDifferentialData.append(np.diff(np.diff(data.iloc[:,0:24].iloc[i].tolist())).max())
        standardDeviation.append(np.std(data.iloc[i]))
  
    featureMatrix['second_differential'] = secondDifferentialData
    featureMatrix['standard_deviation'] = standardDeviation
    return featureMatrix


def main_function():
    data = pd.read_csv('test.csv',header=None)

    dataset = feature_matrix(data)

    with open('trained.pickle', 'rb') as pre_trained:
        pickle_file = load(pre_trained)
        predict = pickle_file.predict(dataset)    
        pre_trained.close()
    
    pd.DataFrame(predict).to_csv('Result.csv',index=False,header=False)

if __name__ == "__main__":
    main_function()