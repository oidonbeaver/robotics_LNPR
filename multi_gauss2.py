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

from scipy.stats import multivariate_normal
# %%
data = pd.read_csv("sensor_data/sensor_data_200.txt",delimiter=" ",header=None,names=("date","time","ir","lidar"))


# %%
d = data.loc[:,["ir","lidar"]]

# %%
sns.jointplot(d["ir"],d["lidar"],d,kind="kde")

# %%
d.cov()

# %%
x,y = np.mgrid[280:340,190:230]

# %%
pos = np.empty(x.shape + (2,))
pos[:,:,0] = x
pos[:,:,1] = y
# %%
irlidar = multivariate_normal(mean = d.mean().values,cov = d.cov().values)

# %%
cont = plt.contour(x,y,irlidar.pdf(pos))
cont.clabel(fmt = "%1.1e")

# %%

x,y = np.mgrid[0:200,0:100]
pos = np.empty(x.shape + (2,))
pos[:,:,0] = x
pos[:,:,1] = y
# %%

a = multivariate_normal(mean=[50,50],cov=[[50,0],[0,100]])
b = multivariate_normal(mean=[100,50],cov=[[125,0],[0,25]])
c = multivariate_normal(mean = [150,50],cov=[[100,-25*(3**0.5)],[-25*(3**0.5),50]])




# %%
for e in [a,b,c]:
    plt.contour(x,y,e.pdf(pos))
plt.gca().set_aspect("equal")
plt.gca().set_xlabel("x")
plt.gca().set_ylabel("y")



# %%
