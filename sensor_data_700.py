#%%
#%%
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm
import random
import seaborn as sns


# %%
data = pd.read_csv("sensor_data/sensor_data_7



00.txt",delimiter=" ",header=None,names=("date","time","ir","lidar"))



# %%
d = data[(data["time"] < 160000)&(data["time"] > 120000) ]
d = d.loc[:,["ir","lidar"]]

# %%
sns.jointplot(d["ir"],d["lidar"],d,kind="kde")

# %%
