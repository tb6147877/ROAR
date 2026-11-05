import time
import threading
from dataclasses import dataclass
from typing import Optional, Any
from agent_base import *



@dataclass
class AgentThreadConfig:
    sleep_when_no_new_obs: float = 0.01
    sleep_before_obs_ready: float = 0.01


class AsyncAgentRunner:
    
    def __init__(
        self,
        agent: BaseAgent,
        obs_buffer,
        action_buffer,
        config: AgentThreadConfig,
    ):
        self.agent = agent
        self.obs_buffer = obs_buffer
        self.action_buffer = action_buffer
        self.config = config

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False

        self._last_obs_version = -1
        self._last_episode_idx = None

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
        counter = 0
        done=False
        
        while not done:
            counter += 1
            packet = self.obs_buffer.read()

            if not packet["ready"]:
                time.sleep(self.config.sleep_before_obs_ready)
                continue

            obs = packet["obs"]
            obs_version = packet["version"]
            info = packet.get("info", {})

            done = packet["done"]

            
            if obs_version == self._last_obs_version:
                time.sleep(self.config.sleep_when_no_new_obs)
                continue

            self._last_obs_version = obs_version

            
            action = self.agent.act(obs)

            
            self.action_buffer.write(action)
        #print("agent stop")

#for testing
class RandomAgent(BaseAgent):
    def __init__(self, name, action_space):
        super().__init__(name)
        self.action_space = action_space

    def act(self, obs):
        return [1,0,1,0,0,0,0]