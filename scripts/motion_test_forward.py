#%%
import sys
sys.path.append("../scripts")
from robot import *
from scipy.stats import multivariate_normal
import copy
##
%matplotlib qt
# %%
world = World(40.0,1,debug=False) 
initial_pose = np.array([0,0,0]).T
robots = []
r = Robot(initial_pose,sensor=None,agent=Agent(0.1,0.0))
# %%
#いっぱいロボットを走らせて描画するだけ
#distance_until_noiseは __init__で決まってしまうので、コピーするごとに変える
for i in range(100):
    copy_r = copy.copy(r)
    copy_r.distance_until_noise = copy_r.noise_pdf.rvs()
    world.append(copy_r)
    robots.append(copy_r)

# %%
world.draw()
# %%
import pandas as pd
# %%
poses = pd.DataFrame([[math.sqrt(r.pose[0]**2+r.pose[1]**2),r.pose[2]] for r in robots],columns=["r","theta"])
# %%
poses.transpose()
# %%
poses["theta"].var()
# %%
poses["r"].mean()
# %%
#p119