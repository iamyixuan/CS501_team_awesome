import numpy as np
import pandas as pd
import torch
import glob

from torch.utils.data import Dataset




class MinMaxScaler:
    def fit(self, x):
        self.min_ = x.min(axis=0)
        self.max_ = x.max(axis=0)
    def transform(self, x):
        x = (x - self.min_) / (self.max_ - self.min_) 
        return x
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)




class WeatherDataset(Dataset):
    """
    Weather dataset class
    Params:
        path: file path
        file_span: number of files to concatenate (one file per month)
        seq_len: sequence length to feed into the RNN
        horizon: steps ahead to predict
    Methods:
        get_files: concatenate selected files
        get_data: prepare sequence instances with corresponding labels
        __len__: return dataset size 
        __getitem__: index samples
    """
    def __init__(self, path, file_span, seq_len, horizon, mode="train") -> None:
        super(WeatherDataset, self).__init__()
        self.file_span = file_span
        self.seq_len = seq_len
        self.horizon = horizon
        self.mode = mode
        

        cloud_cover_dir = path 
        self.cloud_df = self.get_files(cloud_cover_dir + "cloud1_cleaned/")
        self.cloud_df2 = self.get_files(cloud_cover_dir + "cloud2_cleaned/")
        self.cloud_df3 = self.get_files(cloud_cover_dir + "cloud3_cleaned/")

        prcp_dir = path + "FinalPtype/"
        self.prcp = self.get_files(prcp_dir)

        temp_dir = path + "temp_cleaned/"
        self.temp = self.get_files(temp_dir)

        dew_pt_dir = path + "dew_pt_cleaned/"
        self.dew_pt = self.get_files(dew_pt_dir) 

        self.make_data()


    def get_files(self, file_dir):
        filenames = sorted(glob.glob(file_dir + "*.csv"))[:self.file_span]
        dfs = []
        for f in filenames:
            df_tmp = pd.read_csv(f).values[:, 1].reshape(-1, 1) # use the first 10 stations
            dfs.append(df_tmp)
        final_df = np.vstack(dfs)
        return final_df
    
    def get_data(self, df):
        seqs = []
        for i in range(df.shape[0] - self.seq_len - self.horizon): # need to change this later
            for j in range(df.shape[1]):
                tmp = df[i : (i + self.seq_len), j]
                seqs.append(tmp)
        return np.array(seqs)[..., np.newaxis]
    
    def get_label(self, df):
        labels = []
        for i in range(df.shape[0] - self.seq_len - self.horizon): # number of samples to use
            for j in range(df.shape[1]):
                tmp_label = df[(i + self.seq_len + self.horizon), j]
                labels.append(tmp_label)
        labels = np.array(labels)[..., np.newaxis]
        return labels
    
    def make_data(self):
        self.cloud_1 = self.get_data(self.cloud_df)
        self.cloud_2 = self.get_data(self.cloud_df2)
        self.cloud_3 = self.get_data(self.cloud_df3)
        self.temp = self.get_data(self.temp)
        self.dew_pt = self.get_data(self.dew_pt)
        self.prcp_feat = self.get_data(self.prcp)
        self.prcp_feat = np.where(self.prcp_feat >= 1, 1, 0)
        self.prcp = self.get_label(self.prcp)
        self.prcp = np.where(self.prcp >= 1, 1, 0)
        #-------------
        # upsampling the minority class for training
        #------------
        if self.mode == "train":
            _, counts = np.unique(self.prcp, return_counts=True)
            class_len = np.max((counts[0], counts[1]))
            pos_idx = np.unique(np.where(self.prcp == 1)[0])
            pos_over = np.random.choice(pos_idx, (class_len - len(pos_idx)))

            self.cloud_1 = np.vstack([self.cloud_1[pos_over], self.cloud_1])
            self.cloud_2 = np.vstack([self.cloud_2[pos_over], self.cloud_2])
            self.cloud_3 = np.vstack([self.cloud_3[pos_over], self.cloud_3])
            self.temp = np.vstack([self.temp[pos_over], self.temp])
            self.dew_pt = np.vstack([self.dew_pt[pos_over], self.dew_pt])
            self.prcp_feat = np.vstack([self.prcp_feat[pos_over], self.prcp_feat])
            self.prcp = np.vstack([self.prcp[pos_over], self.prcp])
        elif self.mode == "val":
            val_idx = np.random.choice(np.arange(self.prcp.shape[0]), 3000)
            self.cloud_1 = self.cloud_1[val_idx]
            self.cloud_2 = self.cloud_2[val_idx]
            self.cloud_3 = self.cloud_3[val_idx]
            self.temp = self.temp[val_idx]
            self.dew_pt = self.dew_pt[val_idx]
            self.prcp = self.prcp[val_idx] 
            self.prcp_feat = self.prcp_feat[val_idx]

        # df = np.concatenate((self.cloud_1[:, 1], self.cloud_2[:, 1], self.cloud_3[:, 1], self.temp[:,1], self.dew_pt[:,1], self.prcp.reshape(-1, 1)), axis=1)
        # df = pd.DataFrame(df, columns=["c1", "c2", "c3", "temp","dew", "prcp"])
        # df.to_csv("../../Processed Data/train_station_1.csv", index=False)
    def __len__(self):
 
        return self.prcp.shape[0]

    def __getitem__(self, idx):
        Tensor = torch.FloatTensor
        Long = torch.LongTensor
        sample = {"cloud_1": Tensor(self.cloud_1[idx]), "cloud_2": Tensor(self.cloud_2[idx]), "cloud_3": Tensor(self.cloud_3[idx]), 
        "temperature": Tensor(self.temp[idx]), "dew_point": Tensor(self.dew_pt[idx]), "prcp_feat":Tensor(self.prcp_feat[idx]), "precipitation": Tensor(self.prcp[idx])}
        return sample
        
class AustinWeather(Dataset):
    """
    Austin weather from 2013-12-21 to 2017-07-31.
        Input:  
                TempHighF (High temperature, in Fahrenheit)
                TempAvgF (Average temperature, in Fahrenheit)
                TempLowF (Low temperature, in Fahrenheit)
                DewPointHighF (High dew point, in Fahrenheit)
                DewPointAvgF (Average dew point, in Fahrenheit)
                DewPointLowF (Low dew point, in Fahrenheit)
                HumidityHighPercent (High humidity, as a percentage)
                HumidityAvgPercent (Average humidity, as a percentage)
                HumidityLowPercent (Low humidity, as a percentage)
                SeaLevelPressureHighInches (High sea level pressure, in inches)
                SeaLevelPressureAvgInches (Average sea level pressure, in inches)
                SeaLevelPressureLowInches (Low sea level pressure, in inches)
                VisibilityHighMiles (High visibility, in miles)
                VisibilityAvgMiles (Average visibility, in miles)
                VisibilityLowMiles (Low visibility, in miles)
                WindHighMPH (High wind speed, in miles per hour)
                WindAvgMPH (Average wind speed, in miles per hour)
                WindGustMPH (Highest wind speed gust, in miles per hour)
        target: 
                PrecipitationSumInches (Total precipitation, in inches) ('T' if Trace)
                Make labels: put precipitation into 2 classes where if SumInches > 0, assign 1
                            if = 0 assign 0, if 'T', assign 0.
    """
    def __init__(self, datapath, seq_len, horizon,  feat_idx, mode="train"):
        super(AustinWeather, self).__init__()

        feat_idx = [int(i) for i in feat_idx.split(" ")]
        print(feat_idx)
        df = pd.read_csv(datapath).iloc[:, 1:-1]
        df.replace("-", np.nan, inplace=True)
        df.fillna(method="ffill", inplace=True)
        self.seq_len = seq_len
        self.horizon = horizon
        self.mode = mode


        def convert(x):
            if x == 'T' or x == '0':
                return 0
            else:
                return 1

        # split into train and test here (not doing tuning so no val)

        self.scaler = MinMaxScaler()

        train_len = int( 0.8 * df.shape[0])
        train_df = df.iloc[:train_len, :]
        test_df = df.iloc[train_len:, :]

        self.converting = np.vectorize(convert)

        # use scaler for input
        self.scaler.fit(self.get_data(train_df)[..., feat_idx])
        

        if mode == "train":
            self.x = self.get_data(train_df)[..., feat_idx]
            self.y = self.get_label(train_df)
            self.x = self.scaler.transform(self.x)
            
        else:
            self.x = self.get_data(test_df)[..., feat_idx]
            self.y = self.get_label(test_df)       
            self.x = self.scaler.transform(self.x)
        
        self.make_data()

    def get_data(self, df):
        seqs = []
        prcp_hist = self.converting(df.iloc[:, -1]).reshape(-1, 1)
        for i in range(df.shape[0] - self.seq_len - self.horizon): # need to change this later
            tmp = df.iloc[i : (i + self.seq_len), :-1]
            tmp_prcp = prcp_hist[i : (i + self.seq_len), :]
            seqs.append(np.concatenate((tmp, tmp_prcp), axis=-1))
        return np.array(seqs).astype(float)
    
    def get_label(self, df):
        labels = []
        for i in range(df.shape[0] - self.seq_len - self.horizon): # number of samples to use
            tmp_label = df.iloc[(i + self.seq_len + self.horizon), -1]
            converted_label = self.converting(tmp_label)
            labels.append(converted_label)
        labels = np.array(labels)[..., np.newaxis]
        return labels

    def make_data(self):
        #-------------
        # upsampling the minority class for training
        #------------
        if self.mode == "train":
            _, counts = np.unique(self.y, return_counts=True)
            class_len = np.max((counts[0], counts[1]))
            pos_idx = np.unique(np.where(self.y == 1)[0])
            pos_over = np.random.choice(pos_idx, (class_len - len(pos_idx)))

            self.x = np.vstack([self.x[pos_over], self.x])
            self.y = np.vstack([self.y[pos_over], self.y])
        else:
            pass
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        Tensor = torch.FloatTensor
        return Tensor(self.x[idx]), Tensor(self.y[idx])





if __name__ == "__main__":
    d = AustinWeather("../../Processed Data/austin_weather.csv", 12, 5)