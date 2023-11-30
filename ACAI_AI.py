import State
import mmap
import pyvjoy
import time
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import TD3, PPO

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

class ACEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(
            shape=(2,), #steer and gas/brake
            dtype=np.float32, 
            low = -1.0, 
            high = 1.0, 
        )
        self.observation_space = spaces.Box(
            shape=(9,), 
            dtype=np.float32,
            low=np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -MAX_NUM, -MAX_NUM, -MAX_NUM]), 
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, MAX_NUM, MAX_NUM, MAX_NUM]), 
        )
        with open("D:\SteamLibrary\steamapps\common\\assettocorsa\\acai", 'r+b') as f:
            self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        self.curr_state = State.State()
        self.prev_state = None
        self.reset()

    def terminal(self):
        self.episode_end = self.curr_state.isInvalidLap == 1 #if the car goes off track, end the episode
        self.finished = self.curr_state.lapcount > 0

        # print(self.curr_state.isInvalidLap)
        return self.episode_end or self.finished


    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        #series of inputs to reset the game
        vj.set_button(1,1)
        
        time.sleep(1) #required sleep time
        vj.set_button(1,0)

        time.sleep(1)

        vj.set_button(5,1)
        time.sleep(0.5)
        vj.set_button(5,0) #start sim

        vj.data.wAxisX = self.convert_steer_axis(0.0) # center steering
        vj.data.wAxisY = self.convert_axis(0.0) #no throttle
        vj.data.wAxisZ = self.convert_axis(0.5) # half brake to hold in place
        vj.update()

        time.sleep(1)
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
        if (actions[1] >= 0):
            vj.data.wAxisY = self.convert_axis(actions[1])
            vj.data.wAxisZ = self.convert_axis(0.0)
            vj.update()   
        else:
            vj.data.wAxisY = self.convert_axis(0.0)
            vj.data.wAxisZ = self.convert_axis(-actions[1])
            vj.update()

        #get next state
        self.prev_state = self.curr_state
        self.curr_state = self.get_state_shared_mem()
        terminal  = self.terminal()

        if terminal:
            self.reset()

        #convert state to tensorflow format
        next_state = self.convert_state()

        reward = self.reward()
        return next_state, reward, terminal, terminal, {}

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
            track_reward -= 10000000000

        # Reward for speed
        speed_reward= A / (1 + math.exp(-B * (round(self.curr_state.speedKMH) - C)))
        speed_reward *= SPEED_WEIGHT

        # Reward for gap, gap in ms so divide by 100 to reward .1s gap
        # gap_reward = -self.curr_state.gap / 100  * GAP_WEIGHT   # Negative because lower laptime is better.

        track_completion_reward = 0
        if (self.prev_state is not None):
            if round(self.curr_state.normalizedSplinePosition,4) > round(self.prev_state.normalizedSplinePosition,4):
                track_completion_reward = TRACK_COMPLETION_WEIGHT *(round(self.curr_state.normalizedSplinePosition,4) - round(self.prev_state.normalizedSplinePosition,4))
            # else:
            #     track_completion_reward -= TRACK_COMPLETION_WEIGHT * (round(self.prev_state.normalizedSplinePosition,4) - round(self.curr_state.normalizedSplinePosition,4))

        # Combine individual rewards and penalties
        total_reward = track_reward + speed_reward + track_completion_reward

        print("speed reward: ", speed_reward, "track reward: ", track_reward, "track completion reward: ", track_completion_reward, "total reward: ", total_reward, end='\r')

        return total_reward

    def get_state_shared_mem(self):
        time.sleep(0.2) #lets the action occur and psuedo discretize the state
        temp = State.State()
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
        #convert state to tensorforce format
        state_converted = np.array([self.curr_state.speedKMH,
                                    self.curr_state.normalizedSplinePosition,
                                    self.curr_state.gap, 
                                    self.curr_state.rpms, 
                                    self.curr_state.numberOftyresOut, 
                                    #   self.curr_state.laptime, 
                                    self.curr_state.isInvalidLap,
                                    self.curr_state.worldPosition[0],
                                    self.curr_state.worldPosition[1],
                                    self.curr_state.worldPosition[2] 
                                    ], dtype=np.float32) 
        
        #normalize state values from -1 to 1
        state_converted[0] = state_converted[0]/300.0
        state_converted[1] = state_converted[1]/1.0
        state_converted[2] = state_converted[2]/MAX_NUM
        state_converted[3] = state_converted[3]/10000
        state_converted[4] = state_converted[4]/4.0
        state_converted[5] = state_converted[5]/1.0
        # state_converted[6] = state_converted[6]/1000000
        # print(state_converted)


        return state_converted




if __name__ == "__main__":
    env =ACEnv()
    model = PPO("MlpPolicy", env)
    # model.load("ppo_acai.zip")
    model.learn(total_timesteps=1000000)
    model.save("ppo_acai")

    