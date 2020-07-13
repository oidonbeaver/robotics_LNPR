#%%
import matplotlib.pyplot as plt 
import math
import matplotlib.patches as patches
import numpy as np
#%%
import matplotlib
matplotlib.use("nbagg")
import matplotlib.animation as anm
# %matplotlib notebook
%matplotlib qt

# %%
class World:
    def __init__(self):
        self.objects =[]

    def append(self,obj):
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot("111")
        ax.set_aspect("equal")
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        ax.set_xlabel("X",fontsize=20)
        ax.set_ylabel("Y",fontsize=20)
        for obj in self.objects:
            obj.draw(ax)
        plt.show()
        


#%%
class IdealRobot:
    def __init__(self,pose,color="black"):
        self.pose = pose
        self.r = 0.2
        self.color= color
    
    def draw(self,ax):
        x,y,theta = self.pose
        xn = x + self.r*math.cos(theta)
        yn = y + self.r*math.sin(theta)

# %%

world = World()
world.draw()

# %%
