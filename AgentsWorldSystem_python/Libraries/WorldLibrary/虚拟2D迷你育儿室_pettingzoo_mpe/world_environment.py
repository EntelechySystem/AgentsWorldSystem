"""
@Desc   : 虚拟2D迷你育儿室环境
"""

import numpy as np
import pygame
from pettingzoo.utils import ParallelEnv
from pettingzoo.utils import wrappers
from gymnasium import spaces


class MiniVirtualNurseryEnv(ParallelEnv):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self):
        super().__init__()
        self.screen = None
        self.agents = ['婴儿', '教育者']
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(len(self.agents)))))

        self._action_spaces = {
            '婴儿': spaces.Dict({
                '基本的': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),  # 连续动作空间
                '说话': spaces.Text(256),  # 简化地用文本信息模拟语言语音发音，最大发音长度为 256 字符
            }),
            '教育者': spaces.Text(4096)  # 教育者可以发送文本消息模拟教育者的语言语音发音，和发送字词句的视觉信号。最大发送长度为 4096 字符。
        }

        self._observation_spaces = {
            '婴儿': spaces.Dict({
                '视觉': spaces.Box(low=0, high=255, shape=(10, 10, 3), dtype=np.uint8),  # 简化的视觉
                '听觉': spaces.Text(256),  # 简化地用文本信息模拟语言语音听觉，模拟来自教育者发送的认识字词句的视觉信息。最大接收长度为 256 字符。
            }),
            '教育者': spaces.Text(256)  # 教育者可以接收文本消息模拟来自婴儿的语音语言发音信息。最大接收长度为 4096 字符。
        }

        self.reset()

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.baby_position = np.array([5.0, 5.0])
        self.baby_velocity = np.array([0.0, 0.0])
        self.baby_awake = True
        self.baby_eyes_open = True
        self.messages = []
        return self._get_obs()

    def step(self, actions):
        baby_action = actions['婴儿']
        educator_action = actions['教育者']

        self._move_baby(baby_action['基本的'])

        if educator_action == 0:  # 发送消息
            self.messages.append('来自教育者的消息')

        rewards = {agent: 0 for agent in self.agents}
        dones = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return self._get_obs(), rewards, dones, infos

    def _move_baby(self, action):
        self.baby_velocity = action
        self.baby_position += self.baby_velocity

    def _get_obs(self):
        baby_obs = {
            '视觉': np.zeros((10, 10, 3), dtype=np.uint8) if self.baby_eyes_open else np.zeros((10, 10, 3), dtype=np.uint8),
            '听觉': ' '.join(self.messages) if self.baby_awake else ''
        }
        educator_obs = '教育者可以发送消息'
        return {'婴儿': baby_obs, '教育者': educator_obs}

    def observe(self, agent):
        return self._get_obs()[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((500, 500))
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        baby_color = (0, 0, 255) if self.baby_awake else (0, 0, 128)
        pygame.draw.circle(self.screen, baby_color, self.baby_position * 50, 20)

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pass


def main():
    # 测试环境

    # 创建环境实例
    env = MiniVirtualNurseryEnv()
    # env = parallel_to_aec(env)
    # env = wrappers.OrderEnforcingWrapper(env)

    # 重置环境
    observations = env.reset()

    # 运行100步
    for step in range(1):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, dones, infos = env.step(actions)
        env.render()
        print(f"Step {step + 1}")
        print(f"Observations: {observations}")
        print(f"Rewards: {rewards}")
        print(f"Dones: {dones}")
        print(f"Infos: {infos}")
        if all(dones.values()):
            break

    # 关闭环境
    env.close()


if __name__ == "__main__":
    main()
