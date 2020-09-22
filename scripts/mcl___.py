# #%%


import sys
sys.path.append("../scripts/")
from robot import *
from scipy.stats import multivariate_normal
#%%
%matplotlib qt
# # %%

# class EstimationAgent(Agent):
#     """
#     @class EstimationAgent
#     @brief 位置を推定するエージェント
#     """

#     def __init__(self,time_interval,nu,omega,estimator):
#         """
#         #@fn __init__()
#         @brief 説明
#         @param estimator Mclクラスの推定器
#         """
#         super().__init__(nu,omega)
#         self.estimator = estimator#Mclクラス
#         self.time_interval = time_interval
#         self.prev_nu=0.0
#         self.prev_omega=0.0
        
    
#     def draw(self,ax,elems):
#         self.estimator.draw(ax,elems)
#         # elems.append(ax.text(0,0,"hog,fontsize=10e"))
    
#     def decision(self,observation=None):
#         self.estimator.motion_update(self.prev_nu,self.prev_omega,self.time_interval)
#         self.prev_nu,self.prev_omega=self.nu,self.omega#nuとomegaはどこで更新されてる？ｰ>更新されてない、ノイズが逐次加わっているだけ
#         return self.nu,self.omega
# #%%
# class Particle:
#     def __init__(self,init_pose):
#         self.pose = init_pose

#     def motion_update(self,nu,omega,time,noise_rate_pdf):
#         ns = noise_rate_pdf.rvs()
#         noised_nu = nu + ns[0]*math.sqrt(abs(nu)/time)+ ns[1]*math.sqrt(abs(omega)/time)
#         noised_omega = omega + ns[2]*math.sqrt(abs(nu)/time) + ns[3]*math.sqrt(abs(omega)/time)
#         self.pose = IdealRobot.state_transition(noised_nu,noised_omega,time,self.pose)



# #%%
# class Mcl:
#     def __init__(self,init_pose,num,motion_noise_stds):
#         self.particles = [Particle(init_pose) for i in range(num)]

#         v= motion_noise_stds
#         c = np.diag([v["nn"]**2,v["no"]**2,v["on"]**2,v["oo"]**2])
#         self.motion_noise_rate_pdf = multivariate_normal(cov=c)
    
#     def motion_update(self,nu,omega,time):
#         # print(self.motion_noise_rate_pdf.cov)
#         for p in self.particles:
#             p.motion_update(nu,omega,time,self.motion_noise_rate_pdf)
    
#     def draw(self,ax,elems):
#         xs = [p.pose[0] for p in self.particles]#パーティクルのスタート地点 全部同じ座標
#         ys = [p.pose[1] for p in self.particles]
#         vxs = [math.cos(p.pose[2]) for p in self.particles]
#         vys = [math.sin(p.pose[2]) for p in self.particles]
#         elems.append(ax.quiver(xs,ys,vxs,vys,color="blue",alpha=0.5))




#%%
# if __name__ == "main":
# world = World(30,0.1)
# m = Map()
# for ln in [(-4,2),(2,-3),(3,3)]:
#     m.append_landmark(Landmark(*ln))#*はタプルを要素ごとに分ける
# world.append(m)
# # #%%
# # initial_pose = np.array([2,2,math.pi/6]).T
# # estimator = Mcl(initial_pose,100)
# # circling = EstimationAgent(0.2,10.0/180*math.pi,estimator)
# # #%%
# # r = Robot(initial_pose,sensor=Camera(env_map=m),agent=circling)
# # # r = Robot(initial_pose, sensor=Camera(env_map=m,phantom_prob=0.5),agent=circling)
# # world.append(r)
# # # %%
# # world.draw()
# # %%
# initial_pose = np.array([0,0,0]).T
# estimator = Mcl(initial_pose,100,motion_noise_stds={"nn":0.01, "no":0.02,"on":0.03, "oo":0.04})
# a = EstimationAgent(0.1,0.2,10.0/180*math.pi,estimator)
# estimator.motion_update(0.2,10.0/180*math.pi,0.1)
# for p in estimator.particles:
#     print(p.pose)
# %%

# %%
def trial(motion_noise_stds):
    time_interval = 0.1
    world = World(30,time_interval)

    initial_pose = np.array([0,0,0]).T
    estimator=Mcl(initial_pose,100,motion_noise_stds)
    circling = EstimationAgent(time_interval,0.2,10.0/180*math.pi,estimator)
    r = Robot(initial_pose,sensor=None,agent=circling,color="red")
    world.append(r)
    world.draw()


# %%
trial({"nn":0.01,"no":0.02,"on":0.03,"oo":0.04})
# %%

# %%
class Particle: 
    def __init__(self, init_pose):
        self.pose = init_pose
        
    def motion_update(self, nu, omega, time, noise_rate_pdf): 
        ns = noise_rate_pdf.rvs()
        pnu = nu + ns[0]*math.sqrt(abs(nu)/time) + ns[1]*math.sqrt(abs(omega)/time)
        pomega = omega + ns[2]*math.sqrt(abs(nu)/time) + ns[3]*math.sqrt(abs(omega)/time)
        self.pose = IdealRobot.state_transition(pnu, pomega, time, self.pose)