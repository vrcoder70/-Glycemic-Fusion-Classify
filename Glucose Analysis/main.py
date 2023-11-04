import numpy as np 
import pandas as pd 
import datetime
from statistics import mean

def calculate_statistics(dictionary):
    hyperglycemiaList = []
    hyperglycemiaCriticalList = []
    rList = []
    rSList = []
    hypoglycemia1List = []
    hypoglycemia2List = []

    hyperglycemiaDayList = []
    hyperglycemiaCriticalDayList = []
    rDayList = []
    rSDayList = []
    hypoglycemia1DayList = []
    hypoglycemia2DayList = []

    hyperglycemiaNightList = []
    hyperglycemiaCriticalNightList = []
    rNightList = []
    rSNightList = []
    hypoglycemia1NightList = []
    hypoglycemia2NightList = []

    for key, value in dictionary.items():

        #   Calculate for the Whole Day
        hyperglycemia = 0
        hyperglycemiaCritical = 0
        r = 0
        rS = 0
        hypoglycemia1 = 0
        hypoglycemia2 = 0
        
        for v in value[0]:
            if v > 180:
                hyperglycemia += 1
            if v > 250:
                hyperglycemiaCritical += 1
            if v >= 70 and v <= 180:
                r += 1
            if v >= 70 and v <= 150:
                rS += 1
            if v < 70:
                hypoglycemia1 += 1
            if v < 54:
                hypoglycemia2 += 1
        
        hyperglycemiaList.append((hyperglycemia*100)/288)
        hyperglycemiaCriticalList.append((hyperglycemiaCritical*100)/288)
        rList.append((r*100)/288)
        rSList.append((rS*100)/288)
        hypoglycemia1List.append((hypoglycemia1*100)/288) 
        hypoglycemia2List.append((hypoglycemia2*100)/288)
        
        #   Calculate for the Day time
        dayTime = len(value[1])
        if dayTime != 0:
            hyperglycemia = 0
            hyperglycemiaCritical = 0
            r = 0
            rS = 0
            hypoglycemia1 = 0
            hypoglycemia2 = 0
            for v in value[1]:
                if v > 180:
                    hyperglycemia += 1
                if v > 250:
                    hyperglycemiaCritical += 1
                if v >= 70 and v <= 180:
                    r += 1
                if v >= 70 and v <= 150:
                    rS += 1
                if v < 70:
                    hypoglycemia1 += 1
                if v < 54:
                    hypoglycemia2 += 1
            hyperglycemiaDayList.append((hyperglycemia*100)/dayTime)
            hyperglycemiaCriticalDayList.append((hyperglycemiaCritical*100)/dayTime)
            rDayList.append((r*100)/dayTime)
            rSDayList.append((rS*100)/dayTime)
            hypoglycemia1DayList.append((hypoglycemia1*100)/dayTime) 
            hypoglycemia2DayList.append((hypoglycemia2*100)/dayTime)    
        
        #   Calculate for the Night time
        nightTime = len(value[2])
        if nightTime != 0: 
            hyperglycemia = 0
            hyperglycemiaCritical = 0
            r = 0
            rS = 0
            hypoglycemia1 = 0
            hypoglycemia2 = 0
            for v in value[2]:
                if v > 180:
                    hyperglycemia += 1
                if v > 250:
                    hyperglycemiaCritical += 1
                if v >= 70 and v <= 180:
                    r += 1
                if v >= 70 and v <= 150:
                    rS += 1
                if v < 70:
                    hypoglycemia1 += 1
                if v < 54:
                    hypoglycemia2 += 1
            hyperglycemiaNightList.append((hyperglycemia*100)/nightTime)
            hyperglycemiaCriticalNightList.append((hyperglycemiaCritical*100)/nightTime)
            rNightList.append((r*100)/nightTime)
            rSNightList.append((rS*100)/nightTime)
            hypoglycemia1NightList.append((hypoglycemia1*100)/nightTime) 
            hypoglycemia2NightList.append((hypoglycemia2*100)/nightTime)

    stats = [
        mean(hyperglycemiaNightList),
        mean(hyperglycemiaCriticalNightList),
        mean(rNightList),
        mean(rSNightList),
        mean(hypoglycemia1NightList),
        mean(hypoglycemia2NightList),
        
        mean(hyperglycemiaDayList),
        mean(hyperglycemiaCriticalDayList),
        mean(rDayList),
        mean(rSDayList),
        mean(hypoglycemia1DayList),
        mean(hypoglycemia2DayList),
        
        mean(hyperglycemiaList),
        mean(hyperglycemiaCriticalList),
        mean(rList),
        mean(rSList),
        mean(hypoglycemia1List),
        mean(hypoglycemia2List)
    ]

    return stats


def create_segements(cmgData):
    dateSet = set(cmgData['Date'])

    dictionary = {}

    for date in dateSet:
        start = datetime.datetime.strptime(date+'T'+'00:00:00',"%m/%d/%YT%H:%M:%S")
        nightEnd = datetime.datetime.strptime(date+'T'+'05:59:59',"%m/%d/%YT%H:%M:%S")
        dayStart = datetime.datetime.strptime(date+'T'+'06:00:00',"%m/%d/%YT%H:%M:%S")
        end = datetime.datetime.strptime(date+'T'+'23:59:59',"%m/%d/%YT%H:%M:%S")
        
        threeSegments = []
        
        day = []
        dayTime = []
        overNight = []
        for ind in cmgData.index:
            d = cmgData['Date'][ind]
            t = cmgData['Time'][ind]
            currentTimestamp = datetime.datetime.strptime(d+'T'+t,"%m/%d/%YT%H:%M:%S")
            
            if start <= currentTimestamp <= end:
                day.append(cmgData['Sensor Glucose (mg/dL)'][ind])
                
                if currentTimestamp <= nightEnd:
                    overNight.append(cmgData['Sensor Glucose (mg/dL)'][ind])
                else:
                    dayTime.append(cmgData['Sensor Glucose (mg/dL)'][ind])
        
        threeSegments.append(day)
        threeSegments.append(dayTime)
        threeSegments.append(overNight)
        dictionary[date] =threeSegments
    
    return calculate_statistics(dictionary)




def main_function():
    print('Processing Started........')
    cmgPath = '../data/CGMData.csv'
    insulinPath = '../data/InsulinData.csv'
    
    cmg = pd.read_csv(cmgPath)
    insulin = pd.read_csv(insulinPath)

    index = 0
    for ind in insulin.index:
    
        if insulin['Alarm'][ind] == 'AUTO MODE ACTIVE PLGM OFF':
            index = ind
            break
        
    insulinManualMode = insulin.iloc[:index]   
    insulinAutoMode = insulin.iloc[index:] 

    date = insulinAutoMode['Date'][index]
    time = insulinAutoMode['Time'][index]

    autoModeStartTimestamp = datetime.datetime.strptime(date+'T'+time,"%m/%d/%YT%H:%M:%S")

    for ind in cmg.index:
        date = cmg['Date'][ind]
        time = cmg['Time'][ind]

        currentTimestamp = datetime.datetime.strptime(date+'T'+time,"%m/%d/%YT%H:%M:%S")
        
        if currentTimestamp < autoModeStartTimestamp:
            index = ind
            break
    
    cmgManualMode = cmg.iloc[:index-1]   
    cmgAutoMode = cmg.iloc[index-1:] 
    
    # Interpolate NaN values using linear method in forward limit direction
    cmgManualMode = cmgManualMode.interpolate(method ='linear', limit_direction ='forward')
    cmgAutoMode = cmgAutoMode.interpolate(method ='linear', limit_direction ='forward')

    auto = create_segements(cmgAutoMode)
    manual = create_segements(cmgManualMode)

    matrix = [manual,auto]

    df = pd.DataFrame(matrix)
    
    df.to_csv('Result.csv',index=False,header=False)
    
    print('Processing Ended........ Result.csv is created in following folder.')
    print(result+'Result.csv')

if __name__ == "__main__":
    main_function()