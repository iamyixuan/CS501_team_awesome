import pandas as pd
import glob
import numpy as np


cloud1_path = "../../Processed Data/ProcessedDataCLOUD/THIRD/"
prcp_path = "../../Processed Data/FINALTEMP/DEWPOINT/"

c_files = sorted(glob.glob(cloud1_path + "*.csv"))
p_files = sorted(glob.glob(prcp_path + "*.csv"))

for c, p in zip(c_files, p_files):
    print(c[-9:], " ", p[-9:])
    c_df = pd.read_csv(c)
    p_df = pd.read_csv(p)
    print("cloud df shape", c_df.shape, end=" ")
    print("prcp df shape", p_df.shape)
    print()


# print(pd.read_csv(p_files[0]).head())
# print(pd.read_csv(p_files[9]).head())