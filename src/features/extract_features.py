import joblib
import pandas as pd
import logging
from pathlib import Path
from yaml import safe_load
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


# create a logger
logger = logging.getLogger("extract_features")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


def read_cluster_input(data_path, chunksize=100000, usecols=["pickup_latitude","pickup_longitude"]):
    df_reader = pd.read_csv(data_path, chunksize=chunksize, usecols=usecols)
    return df_reader


def save_model(model, save_path):
    joblib.dump(model, save_path)
    

def read_params(params_path="params.yaml"):
    with open(params_path, "r") as file:
        params = safe_load(file)
    return params
    

if __name__ == "__main__":
     # current path
    current_path = Path(__file__)
    # set the root path
    root_path = current_path.parent.parent.parent
    # data_path
    data_path = root_path / "data/interim/df_without_outliers.csv"
    
    # read the data for clustering
    df_reader = read_cluster_input(data_path)
    logger.info("Data read successfully")
    
    # train the standard scaler
    scaler = StandardScaler()
    # train for each chunk
    for chunk in df_reader:
        # fit the scaler
        scaler.partial_fit(chunk)
        
    # save the scaler
    scaler_save_path = root_path / "models/scaler.joblib"
    save_model(scaler, scaler_save_path)
    logger.info("Scaler saved successfully")
    
    # read the data
    df_reader = read_cluster_input(data_path)
    logger.info("Data read successfully")
    
    # read the parameters
    mini_batch_params = read_params()["extract_features"]["mini_batch_kmeans"]
    print("Parameters for clustering are ", mini_batch_params)
    
    # train the kmeans model
    mini_batch = MiniBatchKMeans(**mini_batch_params)
    # train for each chunk
    for chunk in df_reader:
        # scale the chunk
        scaled_chunk = scaler.transform(chunk)
        # train the model
        mini_batch.partial_fit(scaled_chunk)
        
    # save the model
    kmeans_save_path = root_path / "models/mb_kmeans.joblib"
    joblib.dump(mini_batch, kmeans_save_path)
    
    # read the data
    df_final = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"])
    logger.info("Data read for cluster predictions")
    # perform predictions and assign clusters
    location_subset = df_final.loc[:,["pickup_longitude","pickup_latitude"]]
    # scale the input data
    scaled_location_subset = scaler.transform(location_subset)
    # get the cluster predictions
    cluster_predictions = mini_batch.predict(scaled_location_subset)
    
    # save the cluster predictions in data
    df_final['region'] = cluster_predictions
    logger.info("Cluster predictions are added to data")
    # drop the latitude and logitude columns from data
    df_final = df_final.drop(columns=["pickup_latitude","pickup_longitude"])
    logger.info("Latitude and Longitude columns are dropped")
    
    # set the datetime column as index
    df_final.set_index('tpep_pickup_datetime', inplace=True)
    # group the data by region
    region_grp = df_final.groupby("region")
    # resample the data in 15 minute intervals
    resampled_data = (
                        region_grp['region']
                        .resample("15min")
                        .count()
                    )
    logger.info("Data converted to 15 min intervals successfully")
    resampled_data.name = "total_pickups"
    
    # convert back to df
    resampled_data = resampled_data.reset_index(level=0)
    # replace the zeros with an aribitrary value
    epsilon_val = 10
    resampled_data.replace({'total_pickups': {0 : epsilon_val}}, inplace=True)
    
    # read the alpha parameters
    ewma_params = read_params()["extract_features"]["ewma"]
    print("Parameters for EWMA are ", ewma_params)    
    
    # calculate avg pickups using EWMA (shifted by 1 to avoid data leakage)
    resampled_data["avg_pickups"] = (
                resampled_data
                .groupby("region")['total_pickups']
                .ewm(**ewma_params)
                .mean()
                .shift(1)
                .round()
                .values
            )
    logger.info("Average pickups calculated successfully using EWMA")
    
    # save the data
    save_path = root_path / "data/processed/resampled_data.csv"
    resampled_data.to_csv(save_path, index=True)
    logger.info("Data saved successfully")
    