import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any,List
import copy
import gym
import numpy as np

@dataclass
class EnvThreadConfig:
    level_id=0
    step_interval: float = 0.04   # second
    record_video = False
    is_realtime=False

class SharedActionBuffer:
    def __init__(self, default_action):
        self.lock = threading.Lock()
        self.default_action=default_action
        self.action=None

        self.timestamp = 0.0
        self.version = 0

    def write(self, action):
        with self.lock:
            self.action = copy.deepcopy(action)
            self.timestamp = time.time()
            self.version += 1
            #print("act write")

    def read(self):
        with self.lock:
            #print("act read")
            if self.action is None:
                action=copy.deepcopy(self.default_action)
                return action, self.timestamp, self.version
            else:
                action = copy.deepcopy(self.action)
                self.action=None
                return action, self.timestamp, self.version


class SharedObservationBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.obs = None
        self.reward = 0.0
        self.done = False
        self.info = {}
        self.timestamp = 0.0
        self.version = 0
        self.ready = False

    def write(self, obs, reward, done, info):
        with self.lock:
            self.obs =copy.deepcopy(obs)
            self.reward = float(reward)
            self.done = bool(done)
            self.info = info
            self.timestamp = time.time()
            self.version += 1
            self.ready = True
            

    def read(self):
        with self.lock:
            obs = copy.deepcopy(self.obs)
            return {
                "obs": obs,
                "reward": self.reward,
                "done": self.done,
                "info": self.info,
                "timestamp": self.timestamp,
                "version": self.version,
                "ready": self.ready
            }

class AsyncEnvRunner:

    def __init__(
        self,
        env,
        action_buffer: List[SharedActionBuffer],
        obs_buffer: List[SharedObservationBuffer],
        config: EnvThreadConfig = EnvThreadConfig(),
    ):
        self.env = env
        self.action_buffer = action_buffer
        self.obs_buffer = obs_buffer
        self.config = config

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False

        self._episode_step = 0

        self.num_agents = 1
        if self.config.level_id==0:
            self.num_agents = 1

    def start(self):
        if self._started:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._started = True
        return self._thread

    def stop(self):
        if not self._started:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._started = False

    def _run_loop(self):
        obs, info= self.env.reset()
        self._episode_step = 0

        for i in range(len(self.obs_buffer)):
            self.obs_buffer[i].write(
                obs=obs["agent_"+str(i)],
                reward=0.0,
                done=False,
                info=info,
            )

        next_time = time.perf_counter()

        counter=0
        terminate=False
        
        while not terminate:
            counter+=1
            now = time.perf_counter()
            
            if now < next_time:
                time.sleep(next_time - now)
                continue

            actions={}
            for i in range(len(self.action_buffer)):
                action, action_ts, action_ver = self.action_buffer[i].read()
                actions["agent_"+str(i)]=action
            
            obs, reward, done, info = self.env.step(actions)
            self._episode_step += 1

            
            terminate=done['agent_0']

            
            for i in range(len(self.obs_buffer)):
                self.obs_buffer[i].write(
                    obs=obs["agent_" + str(i)],
                    reward=0.0,
                    done=terminate,
                    info=info,
                )

            next_time += self.config.step_interval

            now_after_step = time.perf_counter()
            if next_time < now_after_step:
                next_time = now_after_step

        self.env.close()