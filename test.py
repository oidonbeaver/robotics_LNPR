#%%
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm
import random
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
data.head()

# %%
data["lidar"].hist(bins=max(data["lidar"])-min(data["lidar"]),align="left")

# %%
max(data["lidar"])-min(data["lidar"])


# %%
lidar = data["lidar"]


# %%
mean1 = sum(lidar.values)/len(lidar.values)
mean2 = lidar.mean()
# %%
mean1

# %%
mean2

# %%
lidar.hist(bins=max(lidar)-min(lidar),color="orange",align="left")
plt.vlines(mean1,ymin=0,ymax=5000,color="red")
# %%
zs = lidar.values
mean = sum(zs)/len(zs)
diff_square = [(z-mean)**2 for z in zs]

# %%
sampling_var = sum(diff_square)/(len(zs))
unbiased_var = sum(diff_square)/(len(zs)-1)

# %%

pandas_sampling_var = lidar.var(ddof=False)
pandas_unbiased_var = lidar.var()

# %%
numpy_dafault_var = np.var(lidar)
numpy_unbiased_var = np.var(lidar,ddof=1)

# %%
a= np.array([1,2,3,4,5,5])
stddev1 = math.sqrt(sampling_var)
stddev2 = math.sqrt(unbiased_var)
# %%
a.var()

# %%
freqs = pd.DataFrame(lidar.value_counts())
freqs
# %%
freqs.transpose()

# %%
freqs["probs"] = freqs["lidar"]/len(lidar)

# %%
freqs.transpose()

# %%
sum(freqs["probs"])

# %%
freqs["probs"].sort_index().plot.bar()

# %%
freqs.sample(n=1,weights="probs").index[0]


# %%
def drawing():
    return freqs.sample(n=1,weights="probs").index[0]



# %%
drawing()

# %%

samples = [drawing() for i in range(len(data))]
simulated = pd.DataFrame(samples,columns=["lidar"])
p = simulated["lidar"]
p.hist(bins=max(p)-min(p),color="orange",align="left")

# %%
def p(z,mu=209.7,dev=23.4):
    return math.exp(-(z-mu)**2/(2*dev))/math.sqrt(2*math.pi*dev)


# %%
zs = range(190,230)
ys = [p(z) for z in zs]




# %%
plt.plot(zs,ys)
plt.show()

# %%

def prob(z,width=0.5):
    return width*(p(z-width)+p(z+width))

# %%

plt.bar(zs,ys,color="red",alpha=0.3)
f = freqs["probs"].sort_index()
plt.bar(f.index,f.values,color="blue",alpha=0.3)
plt.show()


# %%
freqs.transpose()

# %%
zs = range(190,230)
ys = [norm.pdf(z,mean1,stddev1) for z in zs]
plt.plot(zs,ys)

# %%
ys = [norm.cdf(z,mean1,stddev1) for z in zs]
plt.plot(zs,ys,color="red")
# %%
ys = [norm.cdf(z+0.5,mean1,stddev1)-norm.cdf(z-0.5,mean1,stddev1) for z in zs]
plt.bar(zs,ys)
# %%
samples = [random.choice([1,2,3,4,5,6]) for i in range(100)]

# %%
