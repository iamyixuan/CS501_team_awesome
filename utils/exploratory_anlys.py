import os
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams["font.size"] = 16
plt.rcParams["lines.linewidth"] = 3

def concat_file(file_path):
    names = sorted(glob.glob(file_path + "*.csv"))
    concat_df = []
    for n in names:
        tmp = pd.read_csv(n).iloc[:, 1:]
        concat_df.append(tmp)
    
    df = pd.concat(concat_df, axis=0)
    return df


def viz_seq(seq, viz_len, seq_name):
    fig, ax = plt.subplots()
    ax.plot(seq[:viz_len])
    ax.set_xlabel("Timestamps")
    ax.set_ylabel(seq_name)
    return fig

#===============
# forward fill faulty values
#===============

def forward_fill(df):
    df = df.replace(9999., np.nan)
    df = df.mask(df > 50, np.nan)
    df = df.mask(df < -40, np.nan)
    df = df.fillna(method="ffill")
    df = df.fillna(method="bfill")
    return df

def make_files(path, mode, file_dir):
    names = sorted(glob.glob(path + "*.csv"))
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    for n in names:
        df = pd.read_csv(n)
        columns = df.columns
        if mode == "temp":
            df = forward_fill(df)
            df = df.drop(df.columns[0], axis=1)
        else:
            df = df.drop(df.columns[0], axis=1)
        df.columns = ["Time"] + [i for i in columns[2:]]
        df.to_csv(file_dir + n[-10:], index=False)



if __name__ == "__main__":
    make_files("../../Processed Data/FINALTEMP/DEWPOINT/", "temp", "../../Processed Data/dew_pt_cleaned/")

    # df = concat_file("../../Processed Data/dew_pt_cleaned/")
    # print(df.describe())





