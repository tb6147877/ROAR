from abc import ABC, abstractmethod
from typing import Optional, Any
from prompt_utils import *
from PIL import Image
import io
import base64
import time
import copy
import numpy as np

class BaseAgent(ABC):
    
    def __init__(self,name):
        self.name = name

    @abstractmethod
    def act(self, obs: Any) -> Any:
        
        pass


    def reset(self):
        
        pass



class LMMBaseAgent(BaseAgent):
    def __init__(self, name, agent_index, level_id, json_format_type, is_realtime):
        super().__init__(name)
        self.is_realtime = is_realtime
        self.agent_index=agent_index
        self.system_prompt=get_system_prompt(self.is_realtime)
        self.task_prompt=get_task_prompt(level_id, self.agent_index)
        self.level_id=level_id
        self.latency_lst=[]

        self.default_response= {
            "movement_x": 0,
            "movement_y": 0,
            "rotation_yaw": 0,
            "rotation_pitch": 0,
            "fire": 0 ,
            "jump": 0 ,
            "minimap": 0
        }

        if json_format_type == 0:
            self.format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "action",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "movement_x": {
                                "type": "integer",
                                "description": "horizontal movement in the agent's local coordinate system, there are three options: 1: move left; 2: move right; 0: no horizontal movement"
                            },
                            "movement_y": {
                                "type": "integer",
                                "description": "vertical movement in the agent's local coordinate system, there are three options: 1: move backward; 2: move forward; 0: no forward/backward movement"
                            },
                            "rotation_yaw": {
                                "type": "integer",
                                "description": "horizontal camera rotation, there are three options: 1: turn left; 2: turn right; 0: no yaw rotation"
                            },
                            "rotation_pitch": {
                                "type": "integer",
                                "description": "vertical camera rotation, there are three options: 1: look down; 2: look up; 0: no pitch rotation"
                            },
                            "fire": {
                                "type": "integer",
                                "description": "there are two options: 1: fire the weapon; 0: do not fire"
                            },
                            "jump": {
                                "type": "integer",
                                "description": "there are two options: 1: jump; 0: do not jump"
                            },
                            "minimap": {
                                "type": "integer",
                                "description": "there are two options: 1: toggle the enlarged minimap; 0: do not toggle the minimap"
                            }
                        },
                        "required": ["movement_x", "movement_y", "rotation_yaw", "rotation_pitch", "fire", "jump", "minimap"],
                        "additionalProperties": False
                    }
                }
            }
        elif json_format_type == 1:
            self.format = {
                "type": "object",
                "properties": {
                            "movement_x": {
                                "type": "integer",
                                "description": "horizontal movement in the agent's local coordinate system, there are three options: 1: move left; 2: move right; 0: no horizontal movement"
                            },
                            "movement_y": {
                                "type": "integer",
                                "description": "vertical movement in the agent's local coordinate system, there are three options: 1: move backward; 2: move forward; 0: no forward/backward movement"
                            },
                            "rotation_yaw": {
                                "type": "integer",
                                "description": "horizontal camera rotation, there are three options: 1: turn left; 2: turn right; 0: no yaw rotation"
                            },
                            "rotation_pitch": {
                                "type": "integer",
                                "description": "vertical camera rotation, there are three options: 1: look down; 2: look up; 0: no pitch rotation"
                            },
                            "fire": {
                                "type": "integer",
                                "description": "there are two options: 1: fire the weapon; 0: do not fire"
                            },
                            "jump": {
                                "type": "integer",
                                "description": "there are two options: 1: jump; 0: do not jump"
                            },
                            "minimap": {
                                "type": "integer",
                                "description": "there are two options: 1: toggle the enlarged minimap; 0: do not toggle the minimap"
                            }
                        },
                        "required": ["movement_x", "movement_y", "rotation_yaw", "rotation_pitch", "fire", "jump", "minimap"]
            }

    def _query(self, prompt, image):
        raise NotImplementedError(self.name + " must implement _query()")

    def _combine_prompts(self, obs):
        result1 = ""
        result1 += self.system_prompt
        result1 += "The following presents general information about the game in JSON format:\n"
        result1 += str(obs["GameState"]) + ",\n"
        result1 += "The following content provides your current status information in JSON format:\n"
        result1 += str(obs["AgentStates"]) + ",\n"
        result1 += "Image observation provided.\n"
        result1 += self.task_prompt

        return result1

    def _case_insensitive_get(self,d, key, default):
        for k in d:
            if k.lower() == key.lower():
                return d[k]
        return default

    def _encode_image(self, img):
        img = Image.fromarray(img)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return base64_image

    def _encode_64_image(self,image):
        return self._encode_image(image)

    def _extract_action(self, action):
        #print(len(action))
        if len(action) != 7:
            return [0,0,0,0,0,0,0]
        a=[]
        a.append(self._case_insensitive_get(action,"movement_x",0))
        a.append(self._case_insensitive_get(action, "movement_y", 0))
        a.append(self._case_insensitive_get(action, "rotation_yaw", 0))
        a.append(self._case_insensitive_get(action, "rotation_pitch", 0))
        a.append(self._case_insensitive_get(action, "fire", 0))
        a.append(self._case_insensitive_get(action, "jump", 0))
        a.append(self._case_insensitive_get(action, "minimap", 0))

        return a

    def print_latency_info(self):
        arr = np.array(self.latency_lst, dtype=float)

        p50 = np.percentile(arr, 50)
        p90 = np.percentile(arr, 90)
        mean = np.mean(arr)

        with open("output.txt", "w", encoding="utf-8") as f:
            f.write("Raw data:\n")
            f.write(", ".join(map(str, arr.tolist())))
            f.write("\n\n")

            f.write("Statistics:\n")
            f.write(f"P50: {p50}\n")
            f.write(f"P90: {p90}\n")
            f.write(f"Mean: {mean}\n")

        return p50, p90, mean

    def act(self, obs):
        if self.is_realtime:
            image_obs = obs["pixel"]
            json_obs = obs["text"]
        else:
            image_obs = obs["agent_" + str(self.agent_index)]["pixel"]
            json_obs = obs["agent_" + str(self.agent_index)]["text"]


        temp = self._combine_prompts(json_obs)
        prompt = temp
        start = time.perf_counter()
        action = self._query(prompt, image_obs)
        #print(action)
        end = time.perf_counter()
        action = self._extract_action(action)
        latency = end - start
        print(action)
        self.latency_lst.append(latency)
        return action


