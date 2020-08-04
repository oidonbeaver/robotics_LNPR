#%%

import matplotlib
matplotlib.use("nbagg")
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np
%matplotlib qt







# %%
class World:        ### fig:world_init_add_timespan (1-6行目)
    def __init__(self, time_span, time_interval, debug=False): #time_span, time_intervalを追加
        self.objects = []  
        self.debug = debug
        self.time_span = time_span                  # 追加
        self.time_interval = time_interval          # 追加
        
    def append(self,obj):             # オブジェクトを登録するための関数
        self.objects.append(obj)
    
    def draw(self):            ### fig:world_draw_with_timespan (11, 22-36行目)
        fig = plt.figure(figsize=(8,8))                # 8x8 inchの図を準備
        ax = fig.add_subplot(111)                      # サブプロットを準備
        ax.set_aspect('equal')                         # 縦横比を座標の値と一致させる
        ax.set_xlim(-5,5)                              # X軸を-5m x 5mの範囲で描画
        ax.set_ylim(-5,5)                              # Y軸も同様に
        ax.set_xlabel("X",fontsize=10)                 # X軸にラベルを表示
        ax.set_ylabel("Y",fontsize=10)                 # 同じくY軸に
        
        elems = []
        
        if self.debug:        
            for i in range(int(self.time_span/self.time_interval)): self.one_step(i, elems, ax)  #固定値から変更
        else:
            ### FuncAnimationのframes, intervalを変更 ###
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax),
            frames=int(self.time_span/self.time_interval)+1, interval=int(self.time_interval*1000), repeat=False)
            plt.show()
        
    def one_step(self, i, elems, ax):
        while elems: elems.pop().remove()
        time_str = "t = %.2f[s]" % (self.time_interval*i)    # 時刻として表示する文字列
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))
        
        for obj in self.objects:
            obj.draw(ax, elems)
            # print(elems)
            if hasattr(obj, "one_step"): obj.one_step(self.time_interval)                 # 変更
        elems.append(ax.text(-4.4, 4, str(len(elems)), fontsize=10))

#%%
class IdealRobot:
    def __init__(self, pose, agent=None, color="black"):
        self.pose = pose        # 引数から姿勢の初期値を設定
        self.r = 0.2            # これは描画のためなので固定値
        self.color = color      # 引数から描画するときの色を設定
        self.agent = agent
        self.poses = [pose] 
    
    def draw(self, ax, elems):
        x, y, theta = self.pose                   # 姿勢の変数を分解して3つの変数へ
        xn = x + self.r * math.cos(theta)         #  ロボットの鼻先のx座標 
        yn = y + self.r * math.sin(theta)         #  ロボットの鼻先のy座標 
        elems += ax.plot([x,xn], [y,yn], color=self.color) # ロボットの向きを示す線分の描画
        # elems = elems + ax.plot([x,xn], [y,yn], color=self.color) # ロボットの向きを示す線分の描画
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color) 
        elems.append(ax.add_patch(c))   # 上のpatches.Circleでロボットの胴体を示す円を作ってサブプロットへ登録
        
        self.poses.append(self.pose)
        # elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black")
        elems = elems + ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black")
        
            
    @classmethod
    def state_transition(self, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array( [nu*math.cos(t0), 
                                     nu*math.sin(t0),
                                     omega ] ) * time
        else:
            return pose + np.array( [nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)), 
                                     nu/omega*(-math.cos(t0 + omega*time) + math.cos(t0)),
                                     omega*time ] )

    def one_step(self, time_interval):
        if not self.agent: return
        nu, omega = self.agent.decision()
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)


#%%

class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega
        
    def decision(self, observation=None):
        return self.nu, self.omega

#%%

world = World(10, 0.1, debug=False)   # 引数を追加         ### fig:add_args_world (1行目だけ)
straight = Agent(0.2, 0.0)                # 0.2[m/s]で直進     
circling = Agent(0.2, 10.0/180*math.pi)   # 0.2[m/s], 10[deg/s]（円を描く
robot1 = IdealRobot( np.array([ 2, 3, math.pi/6]).T,    straight )  
robot2 = IdealRobot( np.array([-2, -1, math.pi/5*6]).T, circling, "red")  
robot3 = IdealRobot( np.array([ 0, 0, 0]).T, color="blue")     # コントローラを与えないロボット
world.append(robot1)
world.append(robot2)
world.append(robot3)
world.draw()


#%%
a=[1]
b=[1]


# %%
a += [2]
b = b + [2]

# %%
a += [3,4]
b = b+ [3,4]

# %%
fig = plt.figure(figsize=(4,4))                # 8x8 inchの図を準備
ax = fig.add_subplot(111)                      # サブプロットを準備
ax.set_aspect('equal')                         # 縦横比を座標の値と一致させる
ax.set_xlim(-5,5)                              # X軸を-5m x 5mの範囲で描画
ax.set_ylim(-5,5)                              # Y軸も同様に
ax.set_xlabel("X",fontsize=10)                 # X軸にラベルを表示
ax.set_ylabel("Y",fontsize=10)                 # 同じくY軸に
#%%
elems = []

# %%
# elems += ax.plot([x,xn], [y,yn], color=self.color) # ロボットの向きを示す線分の描画
elems +=  ax.plot([0,1], [0,1]) # ロボットの向きを示す線分の描画
c = patches.Circle(xy=(0, 0), radius=1, fill=False, color="red") 
elems.append(ax.add_patch(c))   # 上のpatches.Circleでロボットの胴体を示す円を作ってサブプロットへ登録

# %%
elems = elems + ax.plot([0,-1], [0,-1]) # ロボットの向きを示す線分の描画
c = patches.Circle(xy=(2, 2), radius=1, fill=False, color="green") 
elems.append(ax.add_patch(c))   # 上のpatches.Circleでロボットの胴体を示す円を作ってサブプロットへ登録

# %%
 while elems: elems.pop().remove()

# %%
def test(a,b):
    a += [1]
    b = b+[1]

# %%
a=[0]
b=[0]

# %%
test(a,b)

# %%
a

# %%
b

# %%
