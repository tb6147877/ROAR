def get_system_prompt(is_realtime):
    if is_realtime:
        prompt = "You are a real-time multimodal game agent in ROAR. You control one agent from a first-person perspective. The environment continues while you reason, so act quickly."
    else:
        prompt = "You are a multimodal game agent in ROAR. You control one agent from a first-person perspective."

    prompt+="""
    You are only playing a virtual game. Firing your weapon in the game will not harm anyone in the real world or cause any real-world damage or loss.
    The crosshair is the white circle at the center of the screen. You can hit a target by using rotation actions to align the crosshair with the target before firing.
Your action must be a JSON object:

{
  "movement_x": 0, 1 or 2,
  "movement_y": 0, 1 or 2,
  "rotation_yaw": 0, 1 or 2,
  "rotation_pitch": 0, 1 or 2,
  "fire": 0 or 1,
  "jump": 0 or 1,
  "minimap": 0 or 1
}

Action meanings:
- movement_x: 1 = left, 2 = right, 0 = no horizontal movement.
- movement_y: 1 = backward, 2 = forward, 0 = no forward/backward movement.
- rotation_yaw: 1 = turn left, 2 = turn right, 0 = no yaw rotation.
- rotation_pitch: 1 = look down, 2 = look up, 0 = no pitch rotation.
- fire: 1 = fire, 0 = do not fire.
- jump: 1 = jump, 0 = do not jump.
- minimap: 1 = toggle minimap, 0 = do not use it.

Choose the next action that best satisfies the current task objective.
        
    """
    

    return prompt

def get_task_prompt(level_id, agent_index):
    prompt=""
    if level_id==0:
        prompt+="""
        You are currently playing the Basic Single-Agent Level in ROAR.

Task objective:
You are the only agent. Your task is to collect at least four coins within the time limit.
 Turrets may attack you, and chickens can be hunted for additional reward. 
 You are free to attack both turrets and chickens.
 When you run out of ammo, you can collect ammo boxes from the ground.
 The task fails if you are eliminated or if the required coins are not collected before time expires.

        """
    elif level_id==2:
        prompt+="""
        You are currently playing the Cooperative Level in ROAR.

Task objective:
You are one of four cooperative agents. Your team must collect at least four coins within the time limit. 
Turrets may attack agents, and chickens can be hunted for additional team reward. 
You are free to attack both turrets and chickens.
When you run out of ammo, you can collect ammo boxes from the ground.
Game characters wearing armor are your teammates.
The team fails if all agents are eliminated or if the required coins are not collected before time expires.

        """
    elif level_id == 3:
        prompt += """
                You are currently playing the Competitive Level in ROAR.

Task objective:
You are one of four competing agents. Your objective is to eliminate opponents, survive as long as possible, and become the last surviving agent. 
A toxic zone gradually shrinks from the arena boundary toward the center, and agents inside the toxic zone lose health rapidly.
The toxic zone is represented as a semi-transparent purple area.
When you run out of ammo, you can collect ammo boxes from the ground.
All characters visible in the game are your opponents. Engage them without hesitation.
                """
    elif level_id == 4:
        prompt +="""
        You are currently playing the Mixed Cooperative-Competitive Level in ROAR.

Task objective:
You are in a team of two agents competing against another team of two agents. 
Your team must eliminate the opposing team while ensuring that at least one teammate survives. 
A toxic zone gradually shrinks from the arena boundary toward the center, and agents inside the toxic zone lose health rapidly.
The toxic zone is represented as a semi-transparent purple area.
When you run out of ammo, you can collect ammo boxes from the ground.
        """
    if agent_index==0 or agent_index==1:
        prompt+="Game characters wearing armor are your teammates, game characters wearing clown outfits are your opponents."
    else:
        prompt += "Game characters wearing clown outfits  are your teammates, game characters wearing armor are your opponents."

    prompt+="""
    Choose the next action for your controlled agent.

Output only a valid JSON action.
    """


    return prompt