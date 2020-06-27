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
!pwd

# %%
os.listdir()

# %%
os.getcwd()

# %%
data = pd.read_csv("sensor_data/sensor_data_600.txt",delimiter=" ",header=None,names=("date","time","ir","lidar"))


# %%
data["lidar"].hist(bins = max(data["lidar"])-min(data["lidar"]),align="left")
plt.show()

# %%
lidar = data["lidar"]

# %%
lidar.plot()
plt.show()


# %%
data["hour"] = [e//10000 for e in data.time]

# %%
d = data.groupby("hour")
d.lidar.mean().plot()

# %%
d.lidar.get_group(6).hist()
d.lidar.get_group(14).hist()

# %%
each_hour ={i: d.lidar.get_group(i).value_counts().sort_index() for i in range(24) } 

# %%
freqs = pd.concat(each_hour,axis=1)
freqs = freqs.fillna(0)
probs = freqs/len(data)



# %%
sns.heatmap(probs)

# %%
sns.jointplot(data["hour"],data["lidar"],data,kind="kde")

# %%
p_t= pd.DataFrame(probs.sum(axis=0))
p_t.plot()

# %%
p_t.sum()

# %%
p_z = pd.DataFrame(probs.sum(axis=1))

# %%
p_z.plot()

# %%
p_z.transpose()

# %%
p_z.sum()

# %%
p_t.transpose()

# %%
cond_z_t = probs/p_t[0]


# %%
cond_z_t.sum(axis=0)

# %%
cond_z_t[6].plot.bar(color = "blue", alpha = 0.5)
cond_z_t[14].plot.bar(color = "orange", alpha = 0.5)


# %%
cond_t_z = probs.transpose()/probs.transpose().sum()

# %%
cond_t_z = probs.transpose()/probs.sum(axis=1)


# %%
cond_t_z[630][13]

# %%
def bayes_estimation(sensor_value, current_estimation):
    new_estimation = []
    for i in range(24):
        new_estimation.append(cond_z_t[i][sensor_value])
    
    return new_estimation/sum(new_estimation)

# %%
estimation = bayes_estimation(630,p_t[0])
plt.plot(estimation)

# %%
values_5 = [630,632,636]
estimation = p_t[0]
for v in values_5:
    estimation = bayes_estimation(v, estimation)




# %%
plt.plot(estimation)

# %%
