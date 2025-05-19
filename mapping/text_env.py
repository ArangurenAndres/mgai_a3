# import gym
# from gym import spaces
# import numpy as np

# class TextPlatformerEnv(gym.Env):
#     metadata = {"render.modes": ["human"]}

#     def __init__(self, level_file, obs_width=28, obs_height=14):
#         super().__init__()
#         self.obs_width = obs_width
#         self.obs_height = obs_height
#         self.level = self._load_level(level_file)
#         self.height = len(self.level)
#         self.width = len(self.level[0])
#         self.start_pos = self._find_start()
#         self.agent_pos = list(self.start_pos)
#         self.done = False
#         self.cause_of_death = ""
#         self.steps = 0

#         # Actions: 0 = noop, 1 = left, 2 = right, 3 = jump
#         self.action_space = spaces.Discrete(4)
#         self.observation_space = spaces.Box(
#             low=0, high=255, shape=(obs_height, obs_width), dtype=np.uint8
#         )
#         self.vertical_velocity = 0
#         self.gravity = 1

#     def _load_level(self, file_path):
#         with open(file_path, "r") as f:
#             return [list(line.strip()) for line in f.readlines()]

#     def _find_start(self):
#         # Simple start location: first empty tile above solid ground
#         # for y in range(self.height - 2):
#         #     for x in range(self.width):
#         #         if self.level[y][x] == "-" and self.level[y + 1][x] == "X":
#         #             return (x, y)
#         return (1, 12)


#     def get_observation(self):
#         x, y = self.agent_pos
#         half_w = self.obs_width // 2
#         half_h = self.obs_height // 2

#         obs = np.full((self.obs_height, self.obs_width), ord(" "), dtype=np.uint8)

#         for dy in range(-half_h, -half_h + self.obs_height):
#             for dx in range(-half_w, -half_w + self.obs_width):
#                 nx, ny = x + dx, y + dy
#                 if 0 <= nx < self.width and 0 <= ny < self.height:
#                     obs[dy + half_h, dx + half_w] = ord(self.level[ny][nx])

#         return obs


#     def apply_action(self, action):
#         if self.done:
#             return

#         x, y = self.agent_pos

#         # Jump
#         if action == 3 and self._is_on_ground(x, y):
#             self.vertical_velocity = -2

#         # Left / Right
#         if action == 1 and self._is_empty(x - 1, y):
#             x -= 1
#         elif action == 2 and self._is_empty(x + 1, y):
#             x += 1

#         # Gravity
#         self.vertical_velocity += self.gravity
#         for _ in range(abs(self.vertical_velocity)):
#             direction = 1 if self.vertical_velocity > 0 else -1
#             if self._is_empty(x, y + direction):
#                 y += direction
#             else:
#                 self.vertical_velocity = 0
#                 break

#         self.agent_pos = [x, y]

#         # Enemy collision check
#         if self._is_enemy(x, y):
#             if self.vertical_velocity > 0 and self._is_enemy(x, y - 1):  # landed on enemy
#                 self.level[y][x] = '-'  # remove enemy
#                 self.vertical_velocity = -2  # bounce up
#                 self.reward_bonus = 1       # bonus reward
#             else:
#                 self.done = True
#                 self.cause_of_death = "enemy"
#                 self.reward_bonus = -11  # died to enemy
#         else:
#             self.reward_bonus = 0


#     def _is_on_ground(self, x, y):
#         return y + 1 < self.height and self.level[y + 1][x] in ["X","S","?","Q", "<", ">","[", "]"]

#     def _is_empty(self, x, y):
#         return (
#             0 <= x < self.width and 0 <= y < self.height and self.level[y][x] in ["-", "o"] 
#         )

#     def _is_enemy(self, x, y):
#         return 0 <= x < self.width and 0 <= y < self.height and self.level[y][x] == 'E'


#     def calculate_reward(self):
#         x, y = self.agent_pos
#         if y >= self.height:
#             self.done = True
#             self.cause_of_death = "fell off"
#             return -10  # Fell off
#         elif x >= 150 \
#             and x==self.width-1:
#             self.done = True
#             self.cause_of_death = "reached goal"
#             return 1000  # Goal
#         else:
#             return self.reward_bonus  # Progress + any bonus (enemy or penalty)


#     def step(self, action):
#         if self.done:
#             return self.get_observation(), 0, True, {}
        
#         self.steps+=1

#         prev_x = self.agent_pos[0]
#         self.apply_action(action)
#         reward = self.calculate_reward()

#         if self.steps > 500:
#             self.done = True
#             self.cause_of_death = "max steps reached"
#             #print(f"Agent position: {self.agent_pos}, Starting pos: {self.start_pos}")

#         if self.agent_pos[0] > prev_x:
#             reward += 1  # Bonus for moving right
#         if self.agent_pos[0] < prev_x:
#             reward -= 1

#         return self.get_observation(), reward, self.done, {}

    
#     def reset(self):
#         self.agent_pos = list(self.start_pos)
#         self.vertical_velocity = 0
#         self.done = False
#         self.reward_bonus = 0
#         self.steps = 0
#         self.cause_of_death=""
#         return self.get_observation()


#     def render(self, mode="human"):
#         display = [row.copy() for row in self.level]
#         x, y = self.agent_pos
#         if 0 <= y < self.height and 0 <= x < self.width:
#             display[y][x] = "M"
#         print("\n".join("".join(row) for row in display))
#         print()


import gym
from gym import spaces
import numpy as np

class TextPlatformerEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, level_file, obs_width=28, obs_height=14):
        super().__init__()
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.level = self._load_level(level_file)
        self.height = len(self.level)
        self.width = len(self.level[0])
        self.start_pos = self._find_start()
        self.agent_pos = list(self.start_pos)
        self.done = False
        self.cause_of_death = ""
        self.steps = 0
        self.max_x_position = 1  # Track furthest position reached
        
        # Actions: 0 = noop, 1 = left, 2 = right, 3 = jump
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(obs_height, obs_width), dtype=np.uint8
        )
        self.vertical_velocity = 0
        self.gravity = 1

    def _load_level(self, file_path):
        with open(file_path, "r") as f:
            return [list(line.strip()) for line in f.readlines()]

    def _find_start(self):
        # Simple start location: first empty tile above solid ground
        # for y in range(self.height - 2):
        #     for x in range(self.width):
        #         if self.level[y][x] == "-" and self.level[y + 1][x] == "X":
        #             return (x, y)
        return (1, 12)

    def get_observation(self):
        x, y = self.agent_pos
        half_w = self.obs_width // 2
        half_h = self.obs_height // 2

        obs = np.full((self.obs_height, self.obs_width), ord(" "), dtype=np.uint8)

        for dy in range(-half_h, -half_h + self.obs_height):
            for dx in range(-half_w, -half_w + self.obs_width):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    obs[dy + half_h, dx + half_w] = ord(self.level[ny][nx])

        return obs

    def apply_action(self, action):
        if self.done:
            return

        x, y = self.agent_pos

        # Jump
        if action == 3 and self._is_on_ground(x, y):
            self.vertical_velocity = -3  # Increase jump height slightly

        # Left / Right
        if action == 1 and self._is_empty(x - 1, y):
            x -= 1
        elif action == 2 and self._is_empty(x + 1, y):
            x += 1

        # Gravity
        self.vertical_velocity += self.gravity
        for _ in range(abs(self.vertical_velocity)):
            direction = 1 if self.vertical_velocity > 0 else -1
            if self._is_empty(x, y + direction):
                y += direction
            else:
                self.vertical_velocity = 0
                break

        self.agent_pos = [x, y]
        
        # Update max position reached
        if x > self.max_x_position:
            self.max_x_position = x

        # Enemy collision check
        if self._is_enemy(x, y):
            if self.vertical_velocity > 0 and self._is_enemy(x, y - 1):  # landed on enemy
                self.level[y][x] = '-'  # remove enemy
                self.vertical_velocity = -2  # bounce up
                self.reward_bonus = 10       # increased bonus reward
            else:
                self.done = True
                self.cause_of_death = "enemy"
                self.reward_bonus = -20  # Increased penalty for dying to enemy
        else:
            self.reward_bonus = 0

    def _is_on_ground(self, x, y):
        return y + 1 < self.height and self.level[y + 1][x] in ["X","S","?","Q", "<", ">","[", "]"]

    def _is_empty(self, x, y):
        return (
            0 <= x < self.width and 0 <= y < self.height and self.level[y][x] in ["-", "o"] 
        )

    def _is_enemy(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height and self.level[y][x] == 'E'

    def calculate_reward(self):
        x, y = self.agent_pos
        if y >= self.height:
            self.done = True
            self.cause_of_death = "fell off"
            return -50  # Increased penalty for falling
        elif x >= self.width - 5:  # Close to end of level
            self.done = True
            self.cause_of_death = "reached goal"
            return 1000  # Goal
        else:
            # Calculate progress reward based on distance covered
            progress_reward = 0
            if x > self.max_x_position - 3:  # Only if close to or beyond previous max
                progress_reward = 0.5  # Small consistent reward for making progress
                
            # Add checkpoint rewards
            if x > 50 and self.max_x_position <= 50:
                progress_reward += 20  # Checkpoint reward
            if x > 100 and self.max_x_position <= 100:
                progress_reward += 50  # Bigger checkpoint reward
                
            return progress_reward + self.reward_bonus

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, {}
        
        self.steps += 1

        prev_x = self.agent_pos[0]
        self.apply_action(action)
        reward = self.calculate_reward()

        if self.steps > 500:  # Extended max steps
            self.done = True
            self.cause_of_death = "max steps reached"

        # More nuanced movement rewards
        if self.agent_pos[0] > prev_x:
            reward += 0.2  # Small consistent bonus for moving right
        if self.agent_pos[0] < prev_x:
            reward -= 0.5  # Penalty for moving left
            
        # Add info dictionary with useful debugging information
        info = {
            "position": self.agent_pos,
            "max_position": self.max_x_position,
            "steps": self.steps,
            "cause_of_death": self.cause_of_death if self.done else None
        }

        return self.get_observation(), reward, self.done, info
    
    def reset(self):
        self.agent_pos = list(self.start_pos)
        self.vertical_velocity = 0
        self.done = False
        self.reward_bonus = 0
        self.steps = 0
        self.cause_of_death = ""
        self.max_x_position = 1  # Reset the max position tracking
        return self.get_observation()

    def render(self, mode="human"):
        # display = [row.copy() for row in self.level]
        x, y = self.agent_pos
        # if 0 <= y < self.height and 0 <= x < self.width:
        #     display[y][x] = "M"
        #print("\n".join("".join(row) for row in display))
        print(f"Position: ({x}, {y}), Max position: {self.max_x_position}, Steps: {self.steps}")
        print()