#%%
import sys
sys.path.append("../scripts")
from robot import *
from scipy.stats import multivariate_normal
#%%
##
#@class EstimationAgent 
#@brief Mclクラスの推定器,位置を推定して、描画画面に追加する
#@param self.estimator MCLクラスのオブジェクト
##

class EstimationAgent(Agent):
    ##
    #@brief ロボットに命令する移動速度と回転速度の初期化
    #@details ロボットに命令する移動速度と回転速度の初期化,親クラスのメソッドを継承
    ##
    def __init__(self,time_interval,nu,omega,estimator):
        super().__init__(nu,omega)
        self.estimator = estimator
        self.time_interval = time_interval
        self.prev_nu = 0.0
        self.prev_omega = 0.0

    ##
    #@brief Mclの動きを更新する
    #@details (ロボットの代わりにパーティクルを動かす命令を出すイメージ)
    #@return 新しい速度、角速度
    ##
    def decision(self,observation=None):
        self.estimator.motion_update(self.prev_nu, self.prev_omega,self.time_interval)#Mclクラスのmotion_updateメソッドを呼び出す,self.u, self.omega更新
        self.prev_nu, self.prev_omega = self.nu, self.omega#prev_nu,prev_omega更新
        self.estimator.observation_update(observation)
        return self.nu, self.omega

    ##
    #@brief Mclのdraw()を呼び出して描画とelemsへの追加を行う
    ##
    def draw(self,ax,elems):
        # elems.append(ax.text(0,0,"hoge",fontsize=10))
        self.estimator.draw(ax,elems)
#%%
##
#@class Particle
#@brief パーティクル
#@details パーティクル１つ１つを作成
##
class Particle:
    ##
    #@param init_pose np.array([x,y,θ]).T
    ##
    
    def __init__(self,init_pose):
        self.pose = init_pose
    
    ##
    #@brief パーティクルの位置の更新
    #@param noise_rate_pdf nu,omegaに加えるノイズの確率密度関数
    ##
    def motion_update(self,nu,omega,time,noise_rate_pdf):
        ns =noise_rate_pdf.rvs()
        noised_nu = nu + ns[0]*math.sqrt(abs(nu)/time) + ns[1]*math.sqrt(abs(omega)/time)
        noised_omega = omega + ns[2]*math.sqrt(abs(nu)/time) + ns[3]*math.sqrt(abs(omega)/time)
        self.pose = IdealRobot.state_transition(noised_nu,noised_omega,time,self.pose)#パーティクルの新しい姿勢
    
    def observation_update(self,observation):
        print(observation)



#%%
##
#@class Mcl
#@param self.particles num個のパーティクルを作成、更新していく
#@param self.motion_noise_rate_pdf 2x2の確率密度関数
##
class Mcl:
    ##
    #@brief num個のパーティクルをinit_poseの位置で初期化
    #@param init_pose np.array([x,y,θ]).T
    #@param motion_noise_sts 辞書型でnn,no,on,ooのばらつき(標準偏差)を入れる
    ##
    def __init__(self,init_pose, num,motion_noise_stds={"nn":0.19,"no":0.001,"on":0.13,"oo":0.2
    }):

        self.particles = [Particle(init_pose) for i in range(num)]
        v= motion_noise_stds
        c=np.diag([v["nn"]**2,v["no"]**2,v["on"]**2,v["oo"]**2])
        self.motion_noise_rate_pdf = multivariate_normal(cov=c)

    ##
    #@brief Mclの速度、回転速度にノイズを足して次の時刻のMclの位置を作成
    #@details particle 1つ１つに対して行う
    ##
    def motion_update(self,nu,omega,time):
        # print(self.motion_noise_rate_pdf.cov)
        for p in self.particles:
            p.motion_update(nu,omega,time,self.motion_noise_rate_pdf)

    ##
    #@brief 画面にテキストを描き、描画のリスト(elems)にオブジェクトを追加する
    #@param ys particlesのy座標リスト
    #@param xs particlesのx座標リスト
    #@param vxs  particlesの向き(ベクトル)x方向のリスト
    #@param vys particlesの向き(ベクトル)y方向のリスト
    ## 
    def draw(self,ax,elems):

        xs = [p.pose[0] for p in self.particles]#パーティクルのスタート地点 全部同じ座標
        ys = [p.pose[1] for p in self.particles]
        vxs = [math.cos(p.pose[2]) for p in self.particles]
        vys = [math.sin(p.pose[2]) for p in self.particles]
        elems.append(ax.quiver(xs,ys,vxs,vys,color="blue", alpha=0.5))

    def observation_update(self,observation):
        for p in self.particles:
            p.observation_update(observation)

#%%
if __name__ == "__main__":
    # def trial(): ###mcl_obs_prepare
    time_interval = 0.1
    world = World(30, time_interval, debug=True) 

    ### 地図を生成して3つランドマークを追加 ###
    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]: m.append_landmark(Landmark(*ln))
    world.append(m)          

    ### ロボットを作る ###
    initial_pose = np.array([0, 0, 0]).T   #初期位置を原点に
    estimator = Mcl(initial_pose, 100)
    a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, estimator) #EstimationAgentに
    r = Robot(initial_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw()
    
# trial()
# %%
#P116
%matplotlib qt


#%%

if __name__ == "__main__":
    time_interval = 0.1
    world = World(100,time_interval)

    initial_pose = np.array([0,0,0]).T
    estimator=Mcl(initial_pose,100,motion_noise_stds={"nn":0.001,"no":0.001,"on":0.13,"oo":0.001})
    circling = EstimationAgent(time_interval,0.2,10.0/180*math.pi,estimator)
    r = Robot(initial_pose,sensor=None,agent=circling,color="red")
    world.append(r)
    world.draw()

# %%
#p117