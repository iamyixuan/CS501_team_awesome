import numpy as np
import pandas as pd
import glob
import os


def load_data(file_path):
    filenames = os.listdir(file_path)
    for f in filenames:
        df = pd.read_csv(f)
        