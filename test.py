#%%
import pandas as pd
import sys
import os
import  matplotlib.pyplot as plt
# %%
!pwd

# %%
os.listdir()

# %%
os.getcwd()

# %%
data = pd.read_csv("sensor_data/sensor_data_200.txt",delimiter=" ",header=None,names=("date","time","ir","lidar"))



# %%
data

# %%
data["lidar"][0:5]


# %%
