import numpy as np
import pandas as pd
import torch
import glob

from torch.utils.data import Dataset



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
        return labels
    
    def make_data(self):
        self.cloud_1 = self.get_data(self.cloud_df)
        self.cloud_2 = self.get_data(self.cloud_df2)
        self.cloud_3 = self.get_data(self.cloud_df3)
        self.temp = self.get_data(self.temp)
        self.dew_pt = self.get_data(self.dew_pt)
        self.prcp = self.get_label(self.prcp)
        self.prcp = np.where(self.prcp >= 1, 1, 0)
        if self.mode == "trian":
            num_neg = len(self.prcp == 0)
            num_pos = len(self.prcp == 1)
            class_len = np.min((num_neg, num_pos))
            neg_idx = np.where(self.prcp == 0)[0][:class_len]
            pos_idx = np.where(self.prcp == 1)[0][:class_len]
            keep_idx = np.stack((neg_idx, pos_idx))
            keep_idx = np.random.shuffle(keep_idx)

            self.cloud_1 = self.cloud_1[keep_idx]
            self.cloud_2 = self.cloud_2[keep_idx]
            self.cloud_3 = self.cloud_3[keep_idx]
            self.temp = self.temp[keep_idx]
            self.dew_pt = self.dew_pt[keep_idx]
            self.prcp = self.prcp[keep_idx]

    def __len__(self):
 
        return self.prcp.shape[0]

    def __getitem__(self, idx):
        Tensor = torch.FloatTensor
        Long = torch.LongTensor
        sample = {"cloud_1": Tensor(self.cloud_1[idx]), "cloud_2": Tensor(self.cloud_2[idx]), "cloud_3": Tensor(self.cloud_3[idx]), 
        "temperature": Tensor(self.temp[idx]), "dew_point": Tensor(self.dew_pt[idx]), "precipitation": Tensor(self.prcp[idx])}
        return sample
        

if __name__ == "__main__":
    dat = WeatherDataset("../../Processed Data/", 10, 12, 5)
    print(dat[23]["precipitation"].shape)