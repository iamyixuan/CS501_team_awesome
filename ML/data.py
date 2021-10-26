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

        cloud_cover_dir = path + "ProcessedDataCLOUD/"
        self.cloud_df = self.get_files(cloud_cover_dir + "FIRST TYPE/")
        self.cloud_df2 = self.get_files(cloud_cover_dir + "SECOND TYPE/")
        self.cloud_df3 = self.get_files(cloud_cover_dir + "THIRD TYPE/")

        prcp_dir = path + "ProcessedDataPTYPE/"
        self.prcp = self.get_files(prcp_dir)

    def get_files(self, file_dir):
        filenames = sorted(glob.glob(file_dir + "*.csv")[:self.file_span])
        print(filenames)
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
        for i in range(500):
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

        prcp = self.get_label(self.prcp)
        Tensor = torch.FloatTensor
        Long = torch.LongTensor
        sample = {"cloud_1": Tensor(cloud_1[idx]), "cloud_2": Tensor(cloud_2[idx]), " cloud_3": Tensor(cloud_3[idx]), "precipitation": Long(prcp[idx])}
        return sample
        

if __name__ == "__main__":
    dat = WeatherDataset("../../Processed Data/", 3, 12, 5)
    print(dat[23])