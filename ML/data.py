import numpy as np
import pandas as pd
import torch
import glob

from torch.utils.data import Dataset



class WeatherDataset(Dataset):
    def __init__(self, path, file_span, seq_len, horizon) -> None:
        super(WeatherDataset, self).__init__()
        self.file_span = file_span
        self.seq_len = seq_len
        self.horizon = horizon

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


    def get_files(self, file_dir):
        filenames = sorted(glob.glob(file_dir + "*.csv")[:self.file_span])
        dfs = []
        for f in filenames:
            df_tmp = pd.read_csv(f).values[:, :10] # use the first 10 stations
            dfs.append(df_tmp)
        final_df = np.vstack(dfs)
        return final_df
    
    def get_data(self, df):
        seqs = []
        for i in range(500): # need to change this later
            for j in range(1, df.shape[1]):
                tmp = df[i : (i + self.seq_len), j]
                seqs.append(tmp)
        return np.array(seqs)[..., np.newaxis]
    
    def get_label(self, df):
        labels = []
        for i in range(500): # number of samples to use
            for j in range(1, df.shape[1]):
                tmp_label = df[(i + self.seq_len + self.horizon), j]
                labels.append(tmp_label)
        labels = np.array(labels)[..., np.newaxis]
        # label_encoded = np.zeros((labels.shape[0], 6))
        # for l, _ in enumerate(labels):
        #     label_encoded[l, _] = 1
        return labels

    def __len__(self):
        return 500 - self.seq_len - self.horizon

    def __getitem__(self, idx):
        cloud_1 = self.get_data(self.cloud_df)
        cloud_2 = self.get_data(self.cloud_df2)
        cloud_3 = self.get_data(self.cloud_df3)
        temp = self.get_data(self.temp)
        dew_pt = self.get_data(self.dew_pt)
        prcp = self.get_label(self.prcp)
        prcp = np.where(prcp >= 1, 1, 0)
        Tensor = torch.FloatTensor
        Long = torch.LongTensor
        sample = {"cloud_1": Tensor(cloud_1[idx]), "cloud_2": Tensor(cloud_2[idx]), "cloud_3": Tensor(cloud_3[idx]), 
        "temperature": Tensor(temp[idx]), "dew_point": Tensor(dew_pt[idx]), "precipitation": Tensor(prcp[idx])}
        return sample
        

if __name__ == "__main__":
    dat = WeatherDataset("../../Processed Data/", 10, 12, 5)
    print(dat[23]["precipitation"].shape)