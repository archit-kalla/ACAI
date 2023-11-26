from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import State
import mmap

import pyvjoy
import time
# from tensorforce.environments import Environment
# from tensorforce.agents import Agent, ProximalPolicyOptimization, RandomAgent, DeepQNetwork
# from tensorforce.execution import Runner

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment, tf_environment, tf_py_environment, utils, wrappers
from tf_agents.specs import array_spec, tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import py_driver
from tf_agents.utils import common

import reverb

#initialize vjoy controller
vj = pyvjoy.VJoyDevice(1)
# curr_state = State.State()

# def get_state():
#     #get state from localhost:5000
#     data = requests.get('http://localhost:5000/state').json()

#     return data

# weights config

SPEED_WEIGHT = 100
TYRES_OUT_WEIGHT = 1
TYRES_IN_WEIGHT = 1
GAP_WEIGHT = 1.0 #gap is in ms so it is very large
SLIP_ANGLE_WEIGHT = 0.1
TRACK_COMPLETION_WEIGHT = 100
MAX_NUM = 100000000000000000

class ACEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,), #steer and gas/brake
            dtype=np.float32, 
            minimum = -1.0, 
            maximum = 1.0, 
            name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(7,), 
            dtype=np.float32,
            minimum=[0.0, 0.0, -MAX_NUM, 0, 0, 0, 0], 
            maximum=[300.0, 1.0, MAX_NUM, 10000, 4, MAX_NUM, 1], 
            name='observation'
        )

        self.curr_state = State.State()
        self.prev_state = None
    

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def terminal(self):
        self.episode_end = self.curr_state.isInvalidLap == 1 #if the car goes off track, end the episode
        self.finished = self.curr_state.lapcount > 0
        return self.episode_end or self.finished


    def _reset(self):
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

        return ts.restart(state)
    

    def _step(self, actions):
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

        if (self.terminal()):
            return ts.termination(self.convert_state(), self.reward())

        #convert state to tensorflow format
        next_state = self.convert_state()

        reward = self.reward()
        return ts.transition(next_state, reward, discount=1.0)

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

        # Reward for speed
        speed_reward = self.curr_state.speedKMH * SPEED_WEIGHT # You can adjust this based on how much you want to encourage speed.

        # Reward for gap, gap in ms so divide by 100 to reward .1s gap
        gap_reward = -self.curr_state.gap / 100  * GAP_WEIGHT   # Negative because lower laptime is better.

        track_completion_reward = 0
        if (self.prev_state is not None):
            if self.curr_state.normalizedSplinePosition > self.prev_state.normalizedSplinePosition:
                track_completion_reward = 1 * TRACK_COMPLETION_WEIGHT

        # Combine individual rewards and penalties
        total_reward = track_reward + speed_reward + gap_reward + track_completion_reward

        return total_reward

    def get_state_shared_mem(self):
        temp = State.State()
        
        #open shared memory
        with open("C:\Program Files (x86)\Steam\steamapps\common\\assettocorsa\\acai", 'r+b') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            #read shared memory
            mm.seek(0)
            data = mm.read(2048).decode('utf-8')
            # remove bytes after '\0'
            data = data[:data.find('\0')]
            # print(data)

        print(data)    
        
        temp.from_json(data)
        return temp
    

    def convert_state(self):
        self.curr_state
        #convert state to tensorforce format
        state_converted = np.array([self.curr_state.speedKMH,
                          self.curr_state.normalizedSplinePosition,
                          self.curr_state.gap, 
                          self.curr_state.rpms, 
                          self.curr_state.numberOftyresOut, 
                          self.curr_state.laptime, 
                          self.curr_state.isInvalidLap, 
                          ], dtype=np.float32)

        return state_converted
    

def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]




if __name__ == "__main__":
    env =ACEnv()
    
    num_iterations = 250 # @param {type:"integer"}
    collect_episodes_per_iteration = 2 # @param {type:"integer"}
    replay_buffer_capacity = 2000 # @param {type:"integer"}

    fc_layer_params = (100,)
    learning_rate = 1e-3 # @param {type:"number"}
    log_interval = 25 # @param {type:"integer"}
    num_eval_episodes = 10 # @param {type:"integer"}
    eval_interval = 50 # @param {type:"integer"}

    train_env = tf_py_environment.TFPyEnvironment(env)
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        env.observation_spec(),
        env.action_spec(),
        fc_layer_params=fc_layer_params
    )

    train_step_counter = tf.Variable(0)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    tf_agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter)
    tf_agent.initialize()

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy


    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
        tf_agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)
    table = reverb.Table(
        table_name,
        max_size=replay_buffer_capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        tf_agent.collect_data_spec,
        table_name=table_name,
        sequence_length=None,
        local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddEpisodeObserver(
        replay_buffer.py_client,
        table_name,
        replay_buffer_capacity
    )
    def collect_episode(environment, policy, num_episodes):
        driver = py_driver.PyDriver(
            environment,
            py_tf_eager_policy.PyTFEagerPolicy(
            policy, use_tf_function=True),
            [rb_observer],
            max_episodes=num_episodes)
        initial_time_step = environment.reset()
        driver.run(initial_time_step)


    # try:
    #     %%time
    # except:
    #     pass

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    tf_agent.train = common.function(tf_agent.train)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(train_env, tf_agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):

        # # Collect a few episodes using collect_policy and save to the replay buffer.
        # collect_episode(
        #     train_py_env, tf_agent.collect_policy, collect_episodes_per_iteration)

        # Use data from the buffer and update the agent's network.
        iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
        trajectories, _ = next(iterator)
        train_loss = tf_agent.train(experience=trajectories)  

        replay_buffer.clear()

        step = tf_agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(train_env, tf_agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)