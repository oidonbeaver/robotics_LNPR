#%%


import sys
sys.path.append("../scripts/")
from ideal_robot import *
#%%
# %matplotlib qt
#%%
from scipy.stats import  expon, norm, uniform
import copy
# %%


# %%

# %%
##
#@class Robot
#@param self.noise_pdf
#@param self.distance_until_noise
##
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
        obs = self.sensor.data(self.pose) if self.sensor else None#センサ値の雑音が入る
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
        distance_noise_rate=0.1, direction_noise=math.pi/90,\
        distance_bias_rate_stddev=0.1,direction_bias_stddev=math.pi/90,\
        phantom_prob=0.0,phantom_range_x=(-5.0,5.0),phantom_range_y=(-0.5,0.5),\
        oversight_prob=0.1,occulusion_prob=0.0):

        super().__init__(env_map,distance_range,direction_range)
        self.distance_noise_rate = distance_noise_rate
        self.direction_noise = direction_noise
        #バイアスは最初に(オブジェクトを作ったとき)発生する
        self.distance_bias_rate_std = norm.rvs(scale=distance_bias_rate_stddev)
        self.direction_bias=norm.rvs(scale=direction_bias_stddev)
        rx,ry = phantom_range_x, phantom_range_y
        self.phantom_dist = uniform(loc=(rx[0],ry[0]),scale=(rx[1]-rx[0],ry[1]-ry[0]))
        self.phantom_prob=phantom_prob
        self.oversight_prob = oversight_prob
        self.occulusion_prob = occulusion_prob

        
    
    def noise(self, relpos):#relpos 極座標
        ell = norm.rvs(loc=relpos[0],scale=relpos[0]*self.distance_noise_rate)
        phi = norm.rvs(loc=relpos[1],scale=self.direction_noise)
        return np.array([ell,phi]).T

    def bias(self,relpos):
        return relpos + np.array([relpos[0]*self.distance_bias_rate_std,relpos[1]*self.direction_bias]).T

    def phantom(self,cam_pose,relpos):
        if uniform.rvs() < self.phantom_prob:#0から1の一様分布
            #ランドマークの本当の観測(relpos)は使わない
            pos = np.array(self.phantom_dist.rvs()).T #probの確率で半径5mの円の中にファントムが出現
            return self.observation_function(cam_pose,pos)
        else:
            return relpos

    def oversight(self,relpos):
        if uniform.rvs()<self.oversight_prob:
            return None
        else:
            return relpos
    
    def occlusion(self,relpos):
        if uniform.rvs() < self.occulusion_prob:
            ell = relpos[0] + uniform.rvs()*(self.distance_range[1]-relpos[0])
            return np.array(ell,relpos[1]).T
        else:
            return relpos



#観測データの作成。ノイズを入れる
#Robot.one_step()で呼び出される
    def data(self,cam_pose):
        observed=[]
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            z = self.phantom(cam_pose,z)#全く存在しないランドマークが現れるのではなく、実在するランドマークが全然違うところに現れる
            z = self.occlusion(z)
            z = self.oversight(z)#見えなくなる z==Noneになる
            if self.visible(z):
                z=self.bias(z)
                z = self.noise(z)
                
                observed.append((z,lm.id))
        self.lastdata = observed
        return observed


# %%
if __name__ == "__main__":
    world = World(30,0.1,debug=False)
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


