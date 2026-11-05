import requests
import json
from agent_base import *
from const_string import *
from PIL import Image
class ClaudeAndGPTAgent(LMMBaseAgent):

    def __init__(self, name, agent_index, level_id, is_realtime):
        super().__init__(name,agent_index,level_id,0,is_realtime)

    def _query(self, prompt, image):
        #print(prompt)
        arr_256 = image
        if self.level_id == 0:
            img_256 = Image.fromarray(image).resize((256, 256))
            arr_256 = np.array(img_256)
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": LMM_API_KEY,
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": self.name,
                    #"model": "openai/gpt-5.4",
                    "cache_control": {"type": "ephemeral"},
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{self._encode_64_image(arr_256)}"
                                    }
                                }
                            ]
                        },
                    ],
                    "response_format":self.format,
                    "reasoning": {"enabled": False}
                })
            )

            response = response.json()
            #print(response)
            response = json.loads(response['choices'][0]['message']['content'])
            #print(response)
            return response
        except Exception as e:
            print(f"Error：{str(e)}")
            return self.default_response

class GeminiAgent(LMMBaseAgent):

    def __init__(self, agent_index, level_id, is_realtime):
        super().__init__("google/gemini-3.1-pro-preview",agent_index,level_id,0,is_realtime)

    def _query(self, prompt, image):
        #print(prompt)
        arr_256 = image
        if self.level_id == 0:
            img_256 = Image.fromarray(image).resize((256, 256))
            arr_256 = np.array(img_256)
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": LMM_API_KEY,
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": "google/gemini-3.1-pro-preview",
                    #"cache_control": {"type": "ephemeral"},
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{self._encode_64_image(arr_256)}"
                                    }
                                }
                            ]
                        },
                    ],
                    "response_format":self.format,
                    "reasoning": {"effort": "minimal"}
                })
            )

            response = response.json()
            #print(response)
            response = json.loads(response['choices'][0]['message']['content'])
            #print(response)
            return response
        except Exception as e:
            print(f"Error：{str(e)}")
            return self.default_response

class LocalLMMAgent(LMMBaseAgent):

    def __init__(self, name, agent_index, level_id, is_realtime):
        super().__init__(name,agent_index,level_id,1,is_realtime)
        self.name=name
    def _query(self, prompt, image):
        #print(prompt)
        arr_256 = image
        if self.level_id == 0:
            img_256 = Image.fromarray(image).resize((256, 256))
            arr_256 = np.array(img_256)
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.name,
                    "prompt": prompt,
                    "stream": False,
                    "format": self.format,
                    "images": [self._encode_64_image(arr_256)],
                    "options": {"num_ctx": 20000},
                    "think": False
                }
            )

            response = response.json()
            #print(response)
            response = json.loads(response['response'])
            #print(response)
            return response
        except Exception as e:
            print(f"Error：{str(e)}")
            return self.default_response