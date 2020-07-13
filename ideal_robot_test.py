#%%
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np

# %%
import matplotlib
matplotlib.use("nbagg")
import matplotlib.animation as anm
%matplotlib qt

# %%
class World:
    def __init__(self,debug=False):
        self.objects = []
        self.debug = debug

    def append(self, obj):
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        ax.set_xlim(-5,5)
        ax.set_Ylim(-5,5)
        ax.set_xlabel("X",fontsize=20)
        ax.set_ylabel("Y",fontsize=20)
        elems =[]

        if self.debug:
            for i in range(1000):
                self.one_step(i,elems,ax)
    
        else:
            self.ani = anm.FuncAnimation(fig,self.one_step,fargs=(elems,ax),frames = 10,interval = 1000,repeat=False)
            plt.show()

    def one_step(self,i,elems,ax):
        while elems: elems.pop().remove()
        elems.append(ax.text(-4.4,4.5,"t = "+str(i),fontsize=10))
        for obj in self.objects:
            obj.draw(ax,elems)
            if hasattr(obj,"one_step"): obj.one_step(1.0)

#%%
class IdealRobot:
    def __init__(self,pose,agent=None,color="black"):
        self.pose = pose
        self.r = 0.2
        self.color = color
        self.agent = agent
        self.poses = [pose]

    def draw(self,ax,elems):
        x,y,theta = self.pose


