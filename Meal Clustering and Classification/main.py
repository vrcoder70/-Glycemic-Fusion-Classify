import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.fftpack import fft,ifft,rfft
from sklearn.cluster import KMeans,DBSCAN
from scipy.stats import entropy, iqr
from scipy.signal import periodogram
from sklearn.preprocessing import StandardScaler


def get_meal_data(insulin,cmg):
    mealDataTimestamp = []
    for ind in insulin.index:
        date = insulin['Date'][ind] 
        time = insulin['Time'][ind]
        timestamp = datetime.strptime(date+'T'+time,"%m/%d/%YT%H:%M:%S") 
        if mealDataTimestamp:
            previous = mealDataTimestamp[-1]
            if (previous + timedelta(hours=2)) > timestamp:
                mealDataTimestamp.pop()
        mealDataTimestamp.append(timestamp)
    
    glucoseMatrix = []
    insul = []
    for tm in mealDataTimestamp:
        for ind in cmg.index:
            date = cmg['Date'][ind]
            time = cmg['Time'][ind]
            timestamp = datetime.strptime(date+'T'+time,"%m/%d/%YT%H:%M:%S") 
            if timestamp > tm:
                diff = (timestamp - tm).seconds / 60
                if diff > 5:
                    break
                glucose = []
                for i in range(ind+5,ind-25,-1):
                    glucose.append(cmg['Sensor Glucose (mg/dL)'][i])
                
                if not np.isnan(glucose).any():
                    glucoseMatrix.append(glucose)
                    insul.append(tm)
                break
    
    meals = []
    for i in insul:
        for ind in insulin.index:
            date = insulin['Date'][ind] 
            time = insulin['Time'][ind]
            timestamp = datetime.strptime(date+'T'+time,"%m/%d/%YT%H:%M:%S") 

            if i == timestamp:
                meals.append(insulin['BWZ Carb Input (grams)'][ind])
    
    min_val = min(meals)
    max_val = max(meals)
    
    bins = (max_val - min_val)/20
    
    for index,value in enumerate(meals):
        if value < 20:
            meals[index] = 0
        elif 20 <= value < 40:
            meals[index] = 1
        elif 40 <= value < 60:
            meals[index] = 2
        elif 60 <= value < 80:
            meals[index] = 3
        elif 80 <= value < 100:
            meals[index] = 4
        elif 100 <= value < 120:
            meals[index] = 5
        elif 120 <= value:
            meals[index] = 6
        else:
            print('error')

    return pd.DataFrame(glucoseMatrix),pd.DataFrame({'label':meals})

def get_features(meals):
    velocity = meals.diff(axis=1)
    # Calculate min, max, and mean of velocity for each row
    min_vel = velocity.min(axis=1)
    max_vel = velocity.max(axis=1)
    mean_vel = velocity.mean(axis=1)

    data = pd.concat([min_vel, max_vel, mean_vel], axis=1)
    data.columns = ['Min Velocity', 'Max Velocity', 'Mean Velocity']

    acceleration = velocity.diff(axis=1)

    # Calculate min, max, and mean of acceleration for each row
    min_acc = acceleration.min(axis=1)
    max_acc = acceleration.max(axis=1)
    mean_acc = acceleration.mean(axis=1)

    data['Min Acceleration'] = min_acc
    data['Max Acceleration'] = max_acc
    data['Mean Acceleration'] = mean_acc

    entropy_col = meals.apply(entropy, axis=1)

    data['Entropy'] = entropy_col

    iqr_col = meals.apply(iqr, axis=1)

    data['IQR'] = iqr_col

    fft_data = np.abs(np.fft.fft(meals.values, axis=1))

    # Get indices of top 6 max values in each row
    max_indices = np.argsort(-fft_data, axis=1)[:, :6]

    for i in range(6):
        data[f'FFT Max {i+1}'] = fft_data[np.arange(meals.shape[0])[:, None], max_indices][:, i]


    # Apply periodogram to each row of data
    freqs, psd = periodogram(meals.values, axis=1)

    # Create columns for mean PSD values
    data['PSD1'] = psd[:, 0:5].mean(axis=1)
    data['PSD2'] = psd[:, 5:10].mean(axis=1)
    data['PSD3'] = psd[:, 10:16].mean(axis=1)

    return data

def kmean_clustering(data,ground_truth_labels):
    # load the ground truth labels from a CSV file
    clusters = max(ground_truth_labels['label'])
    # normalize the dataframe
    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(data)

    # fit KMeans with 7 clusters
    kmeans = KMeans(n_clusters=clusters+1, random_state=42).fit(df_normalized)

    # get the predicted labels and centroids of the clusters
    predicted_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # calculate the SSE for each cluster
    sse = []
    for i in range(clusters):
        cluster_points = df_normalized[predicted_labels == i]
        sse.append(((cluster_points - centroids[i])**2).sum())

    # calculate the total SSE
    total_sse = sum(sse)

    # calculate the entropy
    entropy = 0
    for i in range(clusters):
        cluster_points = predicted_labels == i
        total_points = len(predicted_labels)
        cluster_size = len(predicted_labels[cluster_points])
        if cluster_size > 0:
            p = cluster_size / total_points
            cluster_labels = ground_truth_labels[cluster_points]
            class_counts = cluster_labels['label'].value_counts()
            class_probs = class_counts / cluster_size
            H = -(class_probs * np.log2(class_probs)).sum()
            entropy += p * H

    # calculate the purity
    purity = 0
    for i in range(clusters):
        cluster_points = predicted_labels == i
        cluster_labels = ground_truth_labels[cluster_points]
        class_counts = cluster_labels['label'].value_counts()
        majority_class_count = class_counts.max()
        cluster_size = len(predicted_labels[cluster_points])
        p = cluster_size / len(predicted_labels)
        purity += p * majority_class_count

    return total_sse,entropy,purity

def dbscan_clustering(data,ground_truth_labels):
    
    # normalize the dataframe
    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(data)

    # fit DBSCAN with eps=1 and min_samples=5
    dbscan = DBSCAN(eps=1, min_samples=5).fit(df_normalized)

    # get the predicted labels
    predicted_labels = dbscan.labels_

    # get the unique labels (excluding noise points)
    clusters = np.unique(predicted_labels[predicted_labels != -1])

    # get the number of clusters (excluding noise points)
    n_clusters = len(clusters)

    # calculate the SSE for each cluster
    sse = []
    for i in range(n_clusters):
        cluster_points = df_normalized[predicted_labels == clusters[i]]
        centroid = np.mean(cluster_points, axis=0)
        sse.append(((cluster_points - centroid)**2).sum())

    # calculate the total SSE
    total_sse = sum(sse)

    # calculate the entropy
    entropy = 0
    for i in range(n_clusters):
        cluster_points = predicted_labels == clusters[i]
        total_points = len(predicted_labels)
        cluster_size = len(predicted_labels[cluster_points])
        if cluster_size > 0:
            p = cluster_size / total_points
            cluster_labels = ground_truth_labels[cluster_points]
            class_counts = cluster_labels['label'].value_counts()
            class_probs = class_counts / cluster_size
            H = -(class_probs * np.log2(class_probs)).sum()
            entropy += p * H

    # calculate the purity
    purity = 0
    for i in range(n_clusters):
        cluster_points = predicted_labels == clusters[i]
        cluster_labels = ground_truth_labels[cluster_points]
        class_counts = cluster_labels['label'].value_counts()
        majority_class_count = class_counts.max()
        cluster_size = len(predicted_labels[cluster_points])
        p = cluster_size / len(predicted_labels)
        purity += p * majority_class_count

    return total_sse,entropy,purity
def main_function():
    print('Process Started')
    insulin_data = pd.read_csv('../data/InsulinData.csv', usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])
    insulin_data = insulin_data.reindex(index=insulin_data.index[::-1])

    insulin_data['BWZ Carb Input (grams)'] = insulin_data['BWZ Carb Input (grams)'].replace(0,pd.NaT)
    insulin_data = insulin_data.dropna()
    insulin_data = insulin_data.reset_index(drop=True)

    cgm_data = pd.read_csv('../data/CGMData.csv',low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)'])
    cgm_data = cgm_data.reindex(index=cgm_data.index[::-1])


    meals,ground_truth = get_meal_data(insulin_data,cgm_data)

    data = get_features(meals)

    kmean_sse,kmean_entropy_val,kmean_purity = kmean_clustering(data,ground_truth)
    
    dbscan_sse,dbscan_entropy_val,dbscan_purity = dbscan_clustering(data,ground_truth)
 

    results = {'col1': [kmean_sse],'col6': [dbscan_sse], 'col2': [kmean_entropy_val], 'col3': [dbscan_entropy_val], 'col4': [kmean_purity], 'col5': [dbscan_purity], }


    df = pd.DataFrame(results)
    df.to_csv('Result.csv',index=False,header=False)
    print('Process Ended')


if __name__ == "__main__":
    main_function()