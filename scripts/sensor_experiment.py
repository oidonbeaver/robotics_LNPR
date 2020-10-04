#%%
import sys
sys.path.append("../scripts")
from robot import *
# %%
m = Map()
m.append_landmark(Landmark(1,0))
# %%
distance =[]
direction=[]
# %%
for i in range(1000):
    c = Camera(m)
    #観測データを作成する。そのときにノイズを入れる座標(0,0),角度0
    d = c.data(np.array([0.0,0.0,0.0]).T)
    if len(d) > 0:
        distance.append(d[0][0][0])
        direction.append(d[0][0][1])

# %%
