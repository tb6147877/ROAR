import gym
import numpy as np
from gym import spaces
import json
import imageio

class RoarMultiAgentWrapper(gym.Wrapper):
    def __init__(self, env, env_info, record_video=False):
        super().__init__(env)

        self.env_info = env_info
        self.num_agents = 1 if env_info["PlayMode"]==0 else 4
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        self.record_video=record_video
        if  record_video:
            self.writer = imageio.get_writer("video.mp4", fps=30)

        if  self.num_agents == 1:
            obs_spaces = {}
            if env_info["EnablePixelObs"] > 0:
                old_space = env.observation_space[0]
                c, h, w = old_space.shape
                obs_spaces["pixel"] = spaces.Box(
                        low=old_space.low.min(),   
                        high=old_space.high.max(), 
                        shape=(h, w, c),
                        dtype=old_space.dtype
                )

            if env_info["EnableDepthObs"] > 0:
                old_space = env.observation_space[1]
                c, h, w = old_space.shape
                obs_spaces["depth"] = spaces.Box(
                        low=old_space.low.min(),   
                        high=old_space.high.max(), 
                        shape=(h, w, c),
                        dtype=old_space.dtype
                )

            if env_info["EnableTextObs"] > 0:
                obs_spaces["text"] = spaces.Dict()

            if env_info["EnableTextObs"] > 0 and env_info["EnableGameStateObs"] > 0: #todo: len needs to confirm
                obs_spaces["game_state"] = spaces.Box(low=-np.inf, high=np.inf,shape=(100,), dtype=np.float32)

            if env_info["EnableTextObs"] > 0 and env_info["EnableAudioObs"] > 0:
                obs_spaces["audio"] = spaces.Dict()

            self.observation_space = spaces.Dict({
                agent: spaces.Dict(obs_spaces)
                for agent in self.agents
            })

            self.action_space = spaces.Dict({
                agent:env.action_space
                for i, agent in enumerate(self.agents)
            })
        else:
            nvec = env.action_space.nvec
            per_agent_act_dim = len(nvec) // self.num_agents
            self.action_space = spaces.Dict({
                agent: spaces.MultiDiscrete(
                    nvec[i * per_agent_act_dim:(i + 1) * per_agent_act_dim]
                )
                for i, agent in enumerate(self.agents)
            })

            obs_spaces = {}
            if env_info["EnablePixelObs"] > 0:
                old_space = env.observation_space[0]
                c, h, w = old_space.shape
                obs_spaces["pixel"] = spaces.Box(
                        low=old_space.low.min(),   
                        high=old_space.high.max(), 
                        shape=(h // 2, w // 2, c),
                        dtype=old_space.dtype
                )

            if env_info["EnableDepthObs"] > 0:
                old_space = env.observation_space[1]
                c, h, w = old_space.shape
                obs_spaces["depth"] = spaces.Box(
                        low=old_space.low.min(),   
                        high=old_space.high.max(), 
                        shape=(h // 2, w // 2, c),
                        dtype=old_space.dtype
                )

            if env_info["EnableTextObs"] > 0:
                obs_spaces["text"] = spaces.Dict()

            if env_info["EnableTextObs"] > 0 and env_info["EnableGameStateObs"] > 0:  # todo: len needs to confirm
                obs_spaces["game_state"] = spaces.Box(low=-np.inf, high=np.inf, shape=(100,), dtype=np.float32)

            if env_info["EnableTextObs"] > 0 and env_info["EnableAudioObs"] > 0:
                obs_spaces["audio"] = spaces.Dict()

            self.observation_space = spaces.Dict({
                agent: spaces.Dict(obs_spaces)
                for agent in self.agents
            })

    def split_image(self, img):
        H, W = img.shape[:2]
        h_mid = H // 2
        w_mid = W // 2

        top_left = img[:h_mid, :w_mid]
        top_right = img[:h_mid, w_mid:]
        bottom_left = img[h_mid:, :w_mid]
        bottom_right = img[h_mid:, w_mid:]

        return top_left, top_right, bottom_left, bottom_right

    def split_obs(self, env_obs):
        if  self.num_agents == 1:
            obs={}
            if self.env_info["EnablePixelObs"] > 0:
                img = env_obs[0].transpose(1, 2, 0)
                img = img.astype(np.float32) / 255.0
                img_gamma = np.power(img, 1.0 / 2.2)
                obs["pixel"] = np.clip(img_gamma * 255.0, 0, 255).astype(np.uint8)
            if self.env_info["EnableDepthObs"] > 0:
                obs["depth"] = env_obs[1].transpose(1,2,0)
            if self.env_info["EnableTextObs"] > 0:
                result = ""
                for code in env_obs[2]:
                    code = int(code)
                    if code == 64: 
                        break
                    result += chr(code)
                obs["text"] = json.loads(result)
            if self.env_info["EnableTextObs"] > 0 and self.env_info["EnableGameStateObs"] > 0:
                print("")# todo:
            if self.env_info["EnableTextObs"] > 0 and self.env_info["EnableAudioObs"] > 0:
                print("")# todo:
            return {self.agents[0]:obs}
        else:
            obs = [{} for _ in range(4)]
            if self.env_info["EnablePixelObs"] > 0:
                img = env_obs[0].transpose(1, 2, 0)
                img = img.astype(np.float32) / 255.0
                img_gamma = np.power(img, 1.0 / 2.2)
                img0, img1,img2, img3 = self.split_image(np.clip(img_gamma * 255.0, 0, 255).astype(np.uint8))
                obs[0]["pixel"] = img0
                obs[1]["pixel"] = img1
                obs[2]["pixel"] = img2
                obs[3]["pixel"] = img3
            if self.env_info["EnableDepthObs"] > 0:
                img0, img1, img2, img3 = self.split_image(env_obs[1].transpose(1,2,0))
                obs[0]["depth"] = img0
                obs[1]["depth"] = img1
                obs[2]["depth"] = img2
                obs[3]["depth"] = img3

            if self.env_info["EnableTextObs"] > 0:
                result = ""
                for code in env_obs[2]:
                    code = int(code)
                    if code == 64:  
                        break
                    result += chr(code)
                data=json.loads(result)
                game_data=data["GameState"]
                agent_data = data["AgentStates"]
                #AgentStates GameState
                obs[0]["text"] = {"GameState":game_data,"AgentStates":[agent_data[0]]}
                obs[1]["text"] = {"GameState":game_data,"AgentStates":[agent_data[1]]}
                obs[2]["text"] = {"GameState":game_data,"AgentStates":[agent_data[2]]}
                obs[3]["text"] = {"GameState":game_data,"AgentStates":[agent_data[3]]}
            if self.env_info["EnableTextObs"] > 0 and self.env_info["EnableGameStateObs"] > 0:
                print("")# todo:
            if self.env_info["EnableTextObs"] > 0 and self.env_info["EnableAudioObs"] > 0:
                print("")# todo:

            return {
                agent: obs[i]
                for i, agent in enumerate(self.agents)
            }


    def merge_action(self, actions):
        return np.concatenate([
            np.asarray(actions[agent], dtype=np.int64)
            for agent in self.agents
        ])

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        obs = result

        if self.record_video:
            img = obs[0].transpose(1, 2, 0)
            img = img.astype(np.float32) / 255.0
            img_gamma = np.power(img, 1.0 / 2.2)
            self.writer.append_data(np.clip(img_gamma * 255.0, 0, 255).astype(np.uint8))

        return self.split_obs(obs), {}

    def step(self, actions):
        single_action = self.merge_action(actions)
        result = self.env.step(single_action)
        obs, reward, done, info = result

        if self.record_video:
            img = obs[0].transpose(1, 2, 0)
            img = img.astype(np.float32) / 255.0
            img_gamma = np.power(img, 1.0 / 2.2)
            self.writer.append_data(np.clip(img_gamma * 255.0, 0, 255).astype(np.uint8))
            if done:
                self.writer.close()

        obs = self.split_obs(obs)

        rewards = {agent: reward for agent in self.agents}
        dones = {agent: done for agent in self.agents}
        infos = {agent: info for agent in self.agents}

        return obs, rewards, dones, infos