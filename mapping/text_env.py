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
        for y in range(self.height - 2):
            for x in range(self.width):
                if self.level[y][x] == "-" and self.level[y + 1][x] == "X":
                    return (x, y)
        return (1, 1)

    # def get_observation(self):
    #     x, y = self.agent_pos
    #     half_w = self.obs_width // 2
    #     half_h = self.obs_height // 2

    #     obs = []
    #     for dy in range(-half_h, half_h + 1):
    #         row = []
    #         for dx in range(-half_w, half_w + 1):
    #             nx, ny = x + dx, y + dy
    #             if 0 <= nx < self.width and 0 <= ny < self.height:
    #                 row.append(ord(self.level[ny][nx]))
    #             else:
    #                 row.append(ord(" "))
    #         obs.append(row)
    #     return np.array(obs, dtype=np.uint8)

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
            self.vertical_velocity = -2

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

                # Enemy collision check
        if self._is_enemy(x, y):
            below_y = y + 1
            if self.vertical_velocity > 0 and self._is_enemy(x, y - 1):  # landed on enemy
                self.level[y][x] = '-'  # remove enemy
                self.vertical_velocity = -2  # bounce up
                self.reward_bonus = 5        # bonus reward
            else:
                self.done = True
                self.reward_bonus = -10  # died to enemy
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
            return -10  # Fell off
        elif self.level[y][x] == "G":
            self.done = True
            return 10  # Goal
        else:
            return 1 + self.reward_bonus  # Progress + any bonus (enemy or penalty)


    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, {}

        prev_x = self.agent_pos[0]
        self.apply_action(action)
        reward = self.calculate_reward()

        if self.agent_pos[0] > prev_x:
            reward += 1  # Bonus for moving right

        return self.get_observation(), reward, self.done, {}

    
    def reset(self):
        self.agent_pos = list(self.start_pos)
        self.vertical_velocity = 0
        self.done = False
        self.reward_bonus = 0
        return self.get_observation()


    def render(self, mode="human"):
        display = [row.copy() for row in self.level]
        x, y = self.agent_pos
        if 0 <= y < self.height and 0 <= x < self.width:
            display[y][x] = "M"
        print("\n".join("".join(row) for row in display))
        print()

