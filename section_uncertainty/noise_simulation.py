#%%


import sys
sys.path.append("../scripts/")
from ideal_robot import *
#%%
%matplotlib qt
#%%
from scipy.stats import  expon, norm, uniform
import copy
# %%


# %%

# %%
class Robot(IdealRobot):
    def __init__(self, pose, agent=None, sensor=None, color="black", \
        noise_per_meter=5, noise_std=math.pi/60,\
        bias_rate_stds=(0.1,0.1),\
        expected_stuck_time=1e100, expected_escape_time=1e-100,\
        expected_kidnap_time=1e100, kidnap_range_x=(-5.0,5.0), kidnap_range_y=(-5.0,5.0)):

        super().__init__(pose,agent,sensor,color)
        self.noise_pdf = expon(scale=1.0/(1e-100+noise_per_meter))#exponは指数分布の関数なので、パラメーターを入れるだけでよい(λを掛けなくてよい)
        self.distance_until_noise = self.noise_pdf.rvs()
        self.theta_noise = norm(scale=noise_std)
        self.bias_rate_nu = norm.rvs(loc=1.0,scale=bias_rate_stds[0])
        self.bias_rate_omega = norm.rvs(loc=1.0,scale=bias_rate_stds[1])

        self.stuck_pdf = expon(scale=expected_stuck_time)#回数の逆数は間隔なのでそのまま入れる
        self.escape_pdf = expon(scale=expected_escape_time)
        self.time_until_stuck = self.stuck_pdf.rvs()
        self.time_until_escape = self.escape_pdf.rvs()
        self.is_stuck = False

        self.kidnap_pdf = expon(scale=expected_kidnap_time)
        self.time_until_kidnap = self.kidnap_pdf.rvs()
        rx, ry =kidnap_range_x, kidnap_range_y
        self.kidnap_dist = uniform(loc=(rx[0],ry[0],0.0),scale=(rx[1]-rx[0],ry[1]-ry[0],2*math.pi))#x,y,theta の3次元


    def noise(self, pose, nu, omega, time_interval):
        #進んだ距離と回転した距離分だけ、ノイズが発生するまでの距離から引く
        self.distance_until_noise -= abs(nu)*time_interval + self.r*abs(omega)*time_interval
        if self.distance_until_noise <= 0.0:
            self.distance_until_noise += self.noise_pdf.rvs()
            #ノイズが発生するとリセット
            #ノイズは姿勢のthetaに加わる
            pose[2] += self.theta_noise.rvs()
        return pose
    
    #コンストラクタ生成時にバイアスの大きさは決まっているので、biasメソッドはその値を掛けるだけ
    def bias(self,nu,omega):
        return nu*self.bias_rate_nu, omega*self.bias_rate_omega

    def stuck(self,nu,omega,time_interval):
        if self.is_stuck:
            self.time_until_escape -= time_interval
            if self.time_until_escape <= 0.0:
                #スタック中に、スタックから脱出したら、脱出までの時間をリセットする
                self.time_until_escape += self.escape_pdf.rvs()
                self.is_stuck = False
        else:
            self.time_until_stuck -= time_interval
            if self.time_until_stuck <= 0.0:
                self.time_until_stuck += self.stuck_pdf.rvs()
                self.is_stuck = True
        
        return nu*(not self.is_stuck), omega*(not self.is_stuck)

    def kidnap(self,pose,time_interval):
        self.time_until_kidnap -= time_interval
        if self.time_until_kidnap <= 0.0:
            self.time_until_kidnap += self.kidnap_pdf.rvs()
            return np.array(self.kidnap_dist.rvs()).T#現在の位置とは関係ないとこに飛ぶ
        else:
            return pose

    


    def one_step(self, time_interval):
        if not self.agent: return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)#ここでワンステップごとにエージェントの命令が復活するのでスタックしても止まったままにならない
        nu, omega = self.bias(nu,omega)#バイアスはエージェントの制御指令に対してかかる
        nu, omega = self.stuck(nu,omega,time_interval)
        self.pose = self.state_transition(nu,omega,time_interval,self.pose)
        self.pose = self.noise(self.pose, nu, omega, time_interval)
        self.pose = self.kidnap(self.pose,time_interval)
        if self.sensor: self.sensor.data(self.pose)
   




#%%
class Camera(IdealCamera):
    def __init__(self, env_map, distance_range=(0.5,6.0),direction_range=(-math.pi/3,math.pi/3),\
        distance_noise_rate=0.1, direction_noise=math.pi/90):
        super().__init__(env_map,distance_range,distance_range)
        self.distance_noise_rate = distance_noise_rate
        self.direction_noise = direction_noise

# %%
world = World(30,0.1)
m = Map()
m.append_landmark(Landmark(-4,2))
m.append_landmark(Landmark(2,-3))
m.append_landmark(Landmark(3,3))
world.append(m)
    
circling = Agent(0.2,10.0/180*math.pi)
r = Robot(np.array([0,0,0]).T, sensor=Camera(env_map=m),agent=circling)
world.append(r)
world.draw()

# %%




# %%


