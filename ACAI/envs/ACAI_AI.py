import mmap
import pyvjoy
import time
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
# from stable_baselines3 import TD3, PPO
# from rl_zoo3 import TD3

#initialize vjoy controller
vj = pyvjoy.VJoyDevice(1)


SPEED_WEIGHT = 1
TYRES_OUT_WEIGHT = 10
TYRES_IN_WEIGHT = 1
GAP_WEIGHT = 1.0 #gap is in ms so it is very large
SLIP_ANGLE_WEIGHT = 0.1
TRACK_COMPLETION_WEIGHT = 10000
MAX_NUM = 1000
A = 100  # Maximum reward
B = 0.1  # Adjust the steepness of the curve
C = 50   # Adjust the horizontal shift
STEER_ANGLE_WEIGHT = 0.5



import json
class State:
    def __init__(self):
        
        self.numberOftyresOut = 0
        self.rpms = 0
        self.isInvalidLap = 0
        self.lapcount = 0
        self.speedKMH = 0
        self.normalizedSplinePosition = 0
        self.gap = 0
        self.laptime = 0
        self.slipAngle = 0
        self.worldPosition = [0,0,0]
        self.velvector = [0,0,0]
        self.carDamaged= 0
        self.steerAngle = 0
        self.distToIdealLine = 0
        self.distToWall_R = 0
        self.distToWall_L = 0

    def from_json(self, json_str):
        json_str = json.loads(json_str)
        self.numberOftyresOut = json_str['numberOftyresOut']
        self.rpms = json_str['rpms']
        self.isInvalidLap = json_str['isInvalidLap']
        self.lapcount = json_str['lapcount']
        self.speedKMH = json_str['speedKMH']
        self.normalizedSplinePosition = json_str['normalizedSplinePosition']
        self.gap = json_str['gap']
        self.laptime = json_str['laptime']
        self.slipAngle = json_str['slipAngle']
        self.worldPosition = json_str['worldPosition']
        self.velvector = json_str['velvector']
        self.carDamaged= json_str['carDamaged']
        self.steerAngle = json_str['steerAngle']
        self.distToIdealLine = json_str['distToIdealLine']
        self.distToWall_R = json_str['distToWall_R']
        self.distToWall_L = json_str['distToWall_L']

        
    



class ACEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(
            shape=(2,), #steer and gas/brake
            dtype=np.float32, 
            low = -1.0, 
            high = 1.0, 
        )
        self.observation_space = spaces.Dict({
            "speedKMH": spaces.Box(low=0, high=300, shape=(1,), dtype=np.float64),
            "normalizedSplinePosition": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64),
            # "gap": spaces.Box(low=0, high=MAX_NUM, shape=(1,), dtype=np.float32),
            "numberOftyresOut": spaces.Box(low=0, high=4, shape=(1,), dtype=np.float64),
            # "laptime": spaces.Box(low=0, high=MAX_NUM, shape=(1,), dtype=np.float32),
            "isInvalidLap": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64),
            "steerAngle": spaces.Box(low=-180, high=180, shape=(1,), dtype=np.float64),
            "worldPosition": spaces.Box(low=-MAX_NUM, high=MAX_NUM, shape=(3,), dtype=np.float64),
            "velvector": spaces.Box(low=-MAX_NUM, high=MAX_NUM, shape=(3,), dtype=np.float64),
            "distToIdealLine": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float64),
            "distToWall_R": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float64),
            "distToWall_L": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float64),
            "slipAngle": spaces.Box(low=-100, high=100, shape=(4,), dtype=np.float64),
        })
        
        
        with open("D:\SteamLibrary\steamapps\common\\assettocorsa\\acai", 'r+b') as f:
            self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        self.curr_state = State()
        self.prev_state = None
        self.reset()

    def terminal(self):
        self.episode_end = self.curr_state.isInvalidLap == 1 or self.curr_state.carDamaged==1 #if the car goes off track, end the episode
        self.finished = self.curr_state.lapcount > 0


        # print(self.curr_state.isInvalidLap)
        return self.episode_end or self.finished


    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        #series of inputs to reset the game
        vj.set_button(1,1)
        
        time.sleep(1) #required sleep time
        vj.set_button(1,0)

        time.sleep(1.5)

        vj.set_button(5,1)
        time.sleep(0.5)
        vj.set_button(5,0) #start sim

        vj.data.wAxisX = self.convert_steer_axis(0.0) # center steering
        vj.data.wAxisY = self.convert_axis(0.0) #no throttle
        vj.data.wAxisZ = self.convert_axis(0.0) # half brake to hold in place
        vj.update()

        time.sleep(1.5)
        self.curr_state = self.get_state_shared_mem()
        # print(self.curr_state)
        self.prev_state = None

        #convert curr_state to tensorflow format
        state=self.convert_state()

        return state, {}
    

    def step(self, actions):
        #send actions to controller
        if (self.terminal()):
            return self.reset()

        vj.data.wAxisX = self.convert_steer_axis(actions[0])
        # print ("steer: ", actions["steer"], "gas_brake: ", actions["gas_brake"])
        vj.data.wAxisY = self.convert_steer_axis(actions[1])
        vj.update()
        # if (actions[1] >= 0):
        #     vj.data.wAxisY = self.convert_axis(actions[1])
        #     vj.data.wAxisZ = self.convert_axis(0.0)
        #     vj.update()   
        # else:
        #     vj.data.wAxisY = self.convert_axis(0.0)
        #     vj.data.wAxisZ = self.convert_axis(-actions[1])
        #     vj.update()

        #get next state
        self.prev_state = self.curr_state
        self.curr_state = self.get_state_shared_mem()
        terminal  = self.terminal()

        if terminal:
            self.reset()

        #convert state to tensorflow format
        next_state = self.convert_state()

        reward = self.reward()
        return next_state, reward, terminal, False, {}

    #convert -1.0 to 1.0 to 0x1 to 0x8000
    def convert_steer_axis(self, value):
        mapped_value = int((value + 1) * 16384)
        return mapped_value
            
    
    #conver 0.0 to 1.0 to 0x1 to 0x8000
    def convert_axis(self, value):
        scaled_value = int(value * 32767)
        return scaled_value
    
    def reward(self):
        track_reward = 0
        if self.curr_state.numberOftyresOut>0:
            track_reward -= self.curr_state.numberOftyresOut * TYRES_OUT_WEIGHT
        else:
            track_reward += 1 * TYRES_IN_WEIGHT
        
        if self.curr_state.isInvalidLap == 1:
            track_reward -= MAX_NUM

        speed_reward = 0
        # # Reward for speed
        # speed_reward= A / (1 + math.exp(-B * (round(self.curr_state.speedKMH) - C)))
        # speed_reward *= SPEED_WEIGHT

        gap_reward = 0
        # Reward for gap, gap in ms so divide by 100 to reward .1s gap
        # gap_reward = -self.curr_state.gap / 100  * GAP_WEIGHT   # Negative because lower laptime is better.

        steer_reward = 0



            
        ideal_line_reward = 0
        wall_R_reward = 0
        wall_L_reward = 0
        track_completion_reward = 0
        if (self.prev_state is not None):
                
            if round(self.curr_state.normalizedSplinePosition,4) > round(self.prev_state.normalizedSplinePosition,4) or (self.curr_state.normalizedSplinePosition < 0.1 and self.prev_state.normalizedSplinePosition > 0.9):
                track_completion_reward = min(TRACK_COMPLETION_WEIGHT *(round(self.curr_state.normalizedSplinePosition,4) - round(self.prev_state.normalizedSplinePosition,4)),10)
                
                if self.curr_state.distToIdealLine <7:
                    ideal_line_reward = 20

                if self.curr_state.distToIdealLine > 10:
                    ideal_line_reward = -5
                
                #lose points for being too close to wall based on distance
                wall_R_reward = -max(10-self.curr_state.distToWall_R,0) *2
                wall_L_reward = -max(10-self.curr_state.distToWall_L,0) *2

                #if 15 from wall, reduce reward
                if self.curr_state.distToWall_R> 15:
                    wall_R_reward = -5
                if self.curr_state.distToWall_L> 15:
                    wall_L_reward = -5

            
            else:
                track_completion_reward = -10 #going backwards is bad

        # # #slip angle reward
        slip_angle_reward = 0
        # # #don't reward slip angle if speed is greater than 3
        # for i in self.curr_state.slipAngle:
        #     if abs(i) > 10 and round(self.curr_state.speedKMH) > 100:
        #         slip_angle_reward -= 10 #out of control

        
            
        # Combine individual rewards and penalties
        if round(self.curr_state.speedKMH)<=3:
            total_reward= 0
        else:
            total_reward = track_reward + speed_reward + track_completion_reward + gap_reward + steer_reward + ideal_line_reward + wall_R_reward + wall_L_reward + slip_angle_reward

        #print rewards for debugging truncating to 2 decimal places
        print("total: {:.2f} on-track: {:.2f} track-comp: {:.2f} speed: {:.2f} gap: {:.2f} steer: {:.2f} ideal: {:.2f} wall_R: {:.2f} wall_L: {:.2f} slip: {:.2f}".format(total_reward,track_reward,track_completion_reward,speed_reward,gap_reward,steer_reward,ideal_line_reward,wall_R_reward,wall_L_reward,slip_angle_reward),end='\r')
        return total_reward

    def get_state_shared_mem(self):
        time.sleep(0.05) #lets the action occur and psuedo discretize the state
        temp = State()
        #read shared memory
        self.mm.seek(0)
        if self.mm.size() < 2048 or self.mm.size() > 2048:
            return self.get_state_shared_mem()
        data = self.mm.read(2048).decode('utf-8')
        # remove bytes after '\0'
        data = data[:data.find('\0')]
        # print(data, end='\r')
        temp.from_json(data)
        return temp


    def convert_state(self):
        #convert state to spaces format
        state_converted = {}
        state_converted["speedKMH"] = np.array([self.curr_state.speedKMH])
        state_converted["normalizedSplinePosition"] = np.array([self.curr_state.normalizedSplinePosition])
        # state_converted["gap"] = np.array([self.curr_state.gap])
        state_converted["numberOftyresOut"] = np.array([self.curr_state.numberOftyresOut/4.0])
        # state_converted["laptime"] = np.array([self.curr_state.laptime])
        state_converted["isInvalidLap"] = np.array([self.curr_state.isInvalidLap/1.0])
        state_converted["steerAngle"] = np.array([self.curr_state.steerAngle])
        state_converted["worldPosition"] = np.array(self.curr_state.worldPosition)
        state_converted["velvector"] = np.array(self.curr_state.velvector)
        state_converted["distToIdealLine"] = np.array([self.curr_state.distToIdealLine])
        state_converted["distToWall_R"] = np.array([self.curr_state.distToWall_R])
        state_converted["distToWall_L"] = np.array([self.curr_state.distToWall_L])
        state_converted["slipAngle"] = np.array(self.curr_state.slipAngle)

        # ret = spaces.Dict(state_converted)
        return state_converted


#if main

if __name__ == "__main__":
    env = ACEnv()
    check_env(env, warn=True)