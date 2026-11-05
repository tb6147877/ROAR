from async_env_runner import *
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
from roar_wrapper import *
from const_string import *
from async_agent_runner import *
from lmm_agents import *

class roar_benchmark:
    def __init__(self, env_config:EnvThreadConfig, agent_config:AgentThreadConfig):
        env_path=""
        env_info_path=""
        if env_config.level_id==0:
            env_path=ROAR_SINGLE_PATH
            env_info_path = ROAR_SINGLE_ENV_CFG_PATH
        elif env_config.level_id==2:
            env_path=ROAR_MULTI_COO_PATH
            env_info_path = ROAR_MULTI_COO_ENV_CFG_PATH
        elif env_config.level_id==3:
            env_path=ROAR_MULTI_COM_PATH
            env_info_path = ROAR_MULTI_COM_ENV_CFG_PATH
        elif env_config.level_id==4:
            env_path=ROAR_MULTI_MIX_PATH
            env_info_path = ROAR_MULTI_MIX_ENV_CFG_PATH

        unity_env = UnityEnvironment(env_path)
        self.env = UnityToGymWrapper(unity_env, uint8_visual=True, allow_multiple_obs=True)

        with open(env_info_path, "r", encoding="utf-8") as f:
            self.envInfo = json.load(f)

        self.env = RoarMultiAgentWrapper(self.env, self.envInfo, env_config.record_video)
        print(self.env.observation_space)
        print(self.env.action_space)

        self.n = self.env.num_agents
        action_buffers =[]
        observation_buffers=[]
        for i in range(self.n):
            action_buffers.append(SharedActionBuffer([0,0,0,0,0,0,0]))
            observation_buffers.append(SharedObservationBuffer())

        self.agents = []
        self.agent_runners=[]
        for i in range(self.n):
            if i<=1:
                a = GeminiAgent(i, env_config.level_id, env_config.is_realtime)
            else:
                #a = ClaudeAndGPTAgent("anthropic/claude-haiku-4.5", i, env_config.level_id, env_config.is_realtime)
                a = ClaudeAndGPTAgent("openai/gpt-5.4", i, env_config.level_id, env_config.is_realtime)
            #a=RandomAgent("random",None)
            #a=ClaudeAndGPTAgent("anthropic/claude-haiku-4.5",i,env_config.level_id,env_config.is_realtime)
            #a = ClaudeAndGPTAgent("openai/gpt-5.4", i, env_config.level_id, env_config.is_realtime)
            #a = GeminiAgent( i, env_config.level_id, env_config.is_realtime)
            #a = LocalLMMAgent("qwen3.5:9b-bf16", i, env_config.level_id, env_config.is_realtime)
            self.agents.append(a)
            self.agent_runners.append(AsyncAgentRunner(a,observation_buffers[i],action_buffers[i],agent_config))

        self.env_runner = AsyncEnvRunner(self.env, action_buffers, observation_buffers, env_config)

    def start_run(self):
        env_t = self.env_runner.start()

        ts=[]
        for i in range(self.n):
            t = self.agent_runners[i].start()
            ts.append(t)

        env_t.join()
        for i in range(self.n):
            ts[i].join()


    def start_run_turn_based(self):
        state,_ = self.env.reset()
        done = False

        counter=0
        while not done:
            counter+=1
            action={}
            for i in range(self.n):
                a = self.agents[i].act(state)
                action["agent_" + str(i)] = a
                #print(a)
            print("-------------")
            state, reward, d,_= self.env.step(action)
            done=d["agent_0"]
            #print(done)

        for i in range(self.n):
            a = self.agents[i].print_latency_info()
            break
        self.env.close()




cfg = EnvThreadConfig()
cfg.level_id=4
cfg.step_interval=0.04
cfg.record_video=True
cfg.is_realtime = False

agent_cfg = AgentThreadConfig()

benchmark = roar_benchmark(cfg,agent_cfg)
#benchmark.start_run()
benchmark.start_run_turn_based()