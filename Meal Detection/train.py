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

def get_meal_data(insulin,cmg,fileIndex):
    mealDataTimestamp = []
    for ind in insulin.index:
        if not math.isnan(insulin['BWZ Carb Input (grams)'][ind]) and insulin['BWZ Carb Input (grams)'][ind] != 0:
            date = insulin['Date'][ind] 
            time = insulin['Time'][ind]
            timestamp = datetime.strptime(date+'T'+time,"%m/%d/%YT%H:%M:%S") if fileIndex == 1 else datetime.strptime(date.split(' ')[0]+'T'+time,"%Y-%m-%dT%H:%M:%S")
            if mealDataTimestamp:
                previous = mealDataTimestamp[-1]
                if (previous + timedelta(hours=2)) > timestamp:
#                     print(previous, '::', timestamp, '::', insulin['BWZ Carb Input (grams)'][ind])
                    mealDataTimestamp.pop()
            mealDataTimestamp.append(timestamp)
    
    glucoseMatrix = []
    for tm in mealDataTimestamp:
        for ind in cmg.index:
            date = cmg['Date'][ind]
            time = cmg['Time'][ind]
            timestamp = datetime.strptime(date+'T'+time,"%m/%d/%YT%H:%M:%S") if fileIndex == 1 else datetime.strptime(date.split(' ')[0]+'T'+time,"%Y-%m-%dT%H:%M:%S")
            if timestamp > tm:
                diff = (timestamp - tm).seconds / 60
#                 print(tm, '::', timestamp)
                if diff > 5:
#                     print('Breaking ',tm, '::', timestamp)
                    break
                glucose = []
                for i in range(ind+5,ind-25,-1):
                    glucose.append(cmg['Sensor Glucose (mg/dL)'][i])
                glucoseMatrix.append(glucose)    
                break
    
    meal_list = []
    for l in glucoseMatrix:
        if not np.isnan(l).any():
            meal_list.append(l)
    return pd.DataFrame (meal_list)

def get_no_meal_data(insulin,cmg,fileIndex):
    noMealDataTimestamp = []
    for ind in insulin.index:
        if math.isnan(insulin['BWZ Carb Input (grams)'][ind]) or insulin['BWZ Carb Input (grams)'][ind] == 0:
            date = insulin['Date'][ind]
            time = insulin['Time'][ind]
            timestamp =  datetime.strptime(date+'T'+time,"%m/%d/%YT%H:%M:%S") if fileIndex == 1 else datetime.strptime(date.split(' ')[0]+'T'+time,"%Y-%m-%dT%H:%M:%S")
            if timestamp not in noMealDataTimestamp:
                noMealDataTimestamp.append(timestamp)
    noMealDataTimestamp  
    
    mealDataTimestampNP = []

    for ind in insulin.index:
        if not math.isnan(insulin['BWZ Carb Input (grams)'][ind]) and insulin['BWZ Carb Input (grams)'][ind] != 0:
            date = insulin['Date'][ind]
            time = insulin['Time'][ind]
            timestamp = datetime.strptime(date+'T'+time,"%m/%d/%YT%H:%M:%S") if fileIndex == 1 else datetime.strptime(date.split(' ')[0]+'T'+time,"%Y-%m-%dT%H:%M:%S")
            mealDataTimestampNP.append(timestamp)
    remove = []
    for tm in mealDataTimestampNP:
        for tn in noMealDataTimestamp:
            if tm <= tn <= (tm + timedelta(hours=2)):
                remove.append(tn)
                
    noMealDataTimestamp = [e for e in noMealDataTimestamp if e not in remove]

    remove = []
    for i in range(len(noMealDataTimestamp)):
        if noMealDataTimestamp[i] not in remove:
            for j in range(i+1, len(noMealDataTimestamp)):
                if noMealDataTimestamp[i] <= noMealDataTimestamp[j] <= noMealDataTimestamp[i]+ timedelta(hours=2):
                    remove.append(noMealDataTimestamp[j])

    noMealDataTimestamp = [e for e in noMealDataTimestamp if e not in remove]
    
    glucoseMatrix = []
    
    for tm in noMealDataTimestamp:
        for ind in cmg.index:
            date = cmg['Date'][ind].split(' ')[0]
            time = cmg['Time'][ind]
            timestamp = datetime.strptime(date+'T'+time,"%m/%d/%YT%H:%M:%S") if fileIndex == 1 else datetime.strptime(date.split(' ')[0]+'T'+time,"%Y-%m-%dT%H:%M:%S")

            if timestamp > tm:
                diff = (timestamp - tm).seconds / 60
                # print(tm, '::', timestamp)
                if diff > 5:
                    # print('Breaking ',tm, '::', timestamp)
                    break
                glucose = []
                for i in range(ind,ind-25,-1):
                    if i >= 0:
                        glucose.append(cmg['Sensor Glucose (mg/dL)'][i])
                glucoseMatrix.append(glucose)    
                break
    no_meal_list = []
    for l in glucoseMatrix:
        if not np.isnan(l).any():
            no_meal_list.append(l)
    
    return pd.DataFrame(no_meal_list)

def meal_features(meals):
    index = meals.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>6).dropna().index
    
    mealData = meals.drop(meals.index[index]).reset_index().drop(columns='index')
    
    indexDrop = mealData.isna().sum(axis=1).replace(0,np.nan).dropna().index
    mealData = mealData.drop(meals.index[indexDrop]).reset_index().drop(columns='index')
   
    mealData = mealData.dropna().reset_index().drop(columns='index')
    
    powerFirstMax = []
    indexFirstMax = []
    powerSecondMax = []
    indexSecondMax = []
    powerThirdMax = []
    
    for i in range(len(mealData)):
        array = abs(rfft(mealData.iloc[:,0:30].iloc[i].values.tolist())).tolist()
        
        sortedArray = abs(rfft(mealData.iloc[:,0:30].iloc[i].values.tolist())).tolist()
        sortedArray.sort()
        
        powerFirstMax.append(sortedArray[-2])
        powerSecondMax.append(sortedArray[-3])
        powerThirdMax.append(sortedArray[-4])
        
        indexFirstMax.append(array.index(sortedArray[-2]))
        indexSecondMax.append(array.index(sortedArray[-3]))
    
    mealFeatureMatrix = pd.DataFrame()
    mealFeatureMatrix['power_second_max'] = powerSecondMax
    mealFeatureMatrix['power_third_max'] = powerThirdMax
   
    tm = mealData.iloc[:,22:25].idxmin(axis=1)
    maximum = mealData.iloc[:,5:19].idxmax(axis=1)
    
    secondDifferentialData = []
    standardDeviation = []
    
    for i in range(len(mealData)):
        secondDifferentialData.append(np.diff(np.diff(mealData.iloc[:,maximum[i]:tm[i]].iloc[i].tolist())).max())
        standardDeviation.append(np.std(mealData.iloc[i]))
 
    mealFeatureMatrix['second_differential']=secondDifferentialData
    mealFeatureMatrix['standard_deviation']=standardDeviation
    return mealFeatureMatrix

def no_meal_features(noMeals):
    index = noMeals.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>5).dropna().index
    
    noMealData=noMeals.drop(noMeals.index[index]).reset_index().drop(columns='index')
    
    indexDrop=noMealData.isna().sum(axis=1).replace(0,np.nan).dropna().index
    noMealData=noMealData.drop(noMealData.index[indexDrop]).reset_index().drop(columns='index')
   
    powerFirstMax=[]
    indexFirstMax=[]
    powerSecondMax=[]
    indexSecondMax=[]
    powerThirdMax=[]
    for i in range(len(noMealData)):
        array=abs(rfft(noMealData.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sortedArray=abs(rfft(noMealData.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sortedArray.sort()
        powerFirstMax.append(sortedArray[-2])
        powerSecondMax.append(sortedArray[-3])
        powerThirdMax.append(sortedArray[-4])
        indexFirstMax.append(array.index(sortedArray[-2]))
        indexSecondMax.append(array.index(sortedArray[-3]))
  
    noMealFeatureMatrix=pd.DataFrame()
    noMealFeatureMatrix['power_second_max']=powerSecondMax
    noMealFeatureMatrix['power_third_max']=powerThirdMax
    
    secondDifferentialData=[]
    standardDeviation=[]
    for i in range(len(noMealData)):
        secondDifferentialData.append(np.diff(np.diff(noMealData.iloc[:,0:24].iloc[i].tolist())).max())
        standardDeviation.append(np.std(noMealData.iloc[i]))
  
    noMealFeatureMatrix['second_differential']=secondDifferentialData
    noMealFeatureMatrix['standard_deviation']=standardDeviation
    return noMealFeatureMatrix

def main_function():
    print('Process Stared........')
   
    insulin_data = pd.read_csv('../data/InsulinData.csv',low_memory=False,usecols=['Date','Time','BWZ Carb Input (grams)'])
    insulin_data1 = pd.read_csv('../data/Insulin_patient2.csv',low_memory=False,usecols=['Date','Time','BWZ Carb Input (grams)'])

    cgm_data = pd.read_csv('../data/CGMData.csv',low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)'])
    cgm_data1 = pd.read_csv('../data/CGM_patient2.csv',low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)'])


    insulin_data = insulin_data.reindex(index=insulin_data.index[::-1])
    insulin_data1 = insulin_data1.reindex(index=insulin_data1.index[::-1])

    cgm_data = cgm_data.reindex(index=cgm_data.index[::-1])
    cgm_data1 = cgm_data1.reindex(index=cgm_data1.index[::-1])

    meal1 = get_meal_data(insulin_data,cgm_data,1)
    meal2 = get_meal_data(insulin_data1,cgm_data1,2)
    meals = pd.concat([meal1, meal2])

    print('Meals data extracted...')

    no_meal1 = get_no_meal_data(insulin_data,cgm_data,1)
    no_meal2 = get_no_meal_data(insulin_data1,cgm_data1,2)
    no_meals = pd.concat([no_meal1, no_meal2])

    print('No Meals data extracted...')

    mealFeatureMatrix = meal_features(meals)
    noMealFeatureMatrix = no_meal_features(no_meals)
    
    mealFeatureMatrix['label']=1
    noMealFeatureMatrix['label']=0
    
    totalData=pd.concat([mealFeatureMatrix,noMealFeatureMatrix]).reset_index().drop(columns='index')
    
    dataset=shuffle(totalData,random_state=1).reset_index().drop(columns='index')
    
    kfold = KFold(n_splits=10,shuffle=False)
    
    unLabeledData=dataset.drop(columns='label')

    scores = []
    model=DecisionTreeClassifier(criterion="entropy")
    
    for train_index, test_index in kfold.split(unLabeledData):
        X_train,X_test,y_train,y_test = unLabeledData.loc[train_index],unLabeledData.loc[test_index],dataset.label.loc[train_index],dataset.label.loc[test_index]
        model.fit(X_train,y_train)
        scores.append(model.score(X_test,y_test))

    classifier=DecisionTreeClassifier(criterion='entropy')
    X,y= unLabeledData, dataset['label']
    classifier.fit(X,y)
    dump(classifier, 'trained.pickle')
    print('Process Ended........')

if __name__ == "__main__":
    main_function()