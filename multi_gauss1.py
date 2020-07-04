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
data = pd.read_csv("sensor_data/sensor_data_700.txt",delimiter=" ",header=None,names=("date","time","ir","lidar"))



# %%
d = data[(data["time"] < 160000)&(data["time"] > 120000) ]
d = d.loc[:,["ir","lidar"]]



# %%
d.ir.var()


# %%
d.lidar.var()

# %%
diff_ir = d.ir - d.ir.mean()
diff_lidar = d.lidar - d.lidar.mean()


# %%
a = diff_lidar * diff_ir

# %%
sum(a)/(len(d)-1)

# %%
sum(a)/(len(a)-1)


# %%
d.mean()

# %%
d

# %%
d.cov()

# %%
d

# %%
irlidar = multivariate_normal(mean=d.mean().values.T,cov = d.cov().values)

# %%
multivariate_normal(mean=d.mean().values.T,cov = d.cov())
multivariate_normal(mean=d.mean(),cov = d.cov())


# %%
x,y = np.mgrid[0:40,710:750]
pos = np.empty(x.shape+(2,))
pos[:,:,0] = x
pos[:,:,1] = y

# %%
cont = plt.contour(x,y,irlidar.pdf(pos))
cont.clabel(fmt= "%1.1e")

# %%

c = d.cov().values + np.array([[0,20],[20,0]])
tmp = multivariate_normal(mean = d.mean(),cov = c)
cont = plt.contour(x,y,tmp.pdf(pos))
plt.show()
# %%
