"""
@Desc   : 虚拟2D迷你育儿室环境 Virtual 2D Mini Nursery Env，基于 PettingZoo 的自行改造的 MPE 环境。
action_spaces = {
    '说话': spaces.Text(256),  # 简化地用文本信息模拟语言语音发音，单次最大发音长度为指定的字符
    '移动运动': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),  # 连续动作空间
    '表情': spaces.Discrete(6),  # 0: 静，1: 喜，2: 怒，3: 哀，4: 惧，5: 思。 这里简化了输出的表情为六元动作。
    '抓取运动': spaces.Discrete(2),  # 0: 无抓取物体，1: 抓取物体 。这里简化抓取运动为二元动作。真实的抓取运动十分复杂，需要更复杂的动作空间。
    '睡眠': spaces.Discrete(2),  # 0: 醒来，1: 睡觉。这里简化睡眠状态为二元动作。
    '饮食': spaces.Discrete(2),  # 0: 未进食，1: 进食。这里简化饥饿状态为二元动作。
}
observation_spaces = {
    '看到的内容': spaces.Box(low=0, high=255, shape=(640, 480, 3), dtype=np.uint8),  # 简化的视觉。以传入的图像信息模拟视觉信息。最大接收长度为 640x480x3 像素通道。通过多个时刻接收的图像作为帧，作为接收的视频信息。
    '听到的内容': spaces.MultiDiscrete([256] * 256)  # 简化地用编码的文本信息模拟语言语音听觉，模拟来自教育者发送的认识字词句的文本信息。最大接收长度为指定的编码后的数组长度
    '呈现的表情': spaces.Discrete(6),  # 0: 静，1: 喜，2: 怒，3: 哀，4: 惧，5: 思。 这里简化了输出的表情为离散动作值
    '摸到的内容': spaces.Discrete(5),  # 0: 无碰触，1: 轻度碰触，2: 中度碰触，3: 重度碰触，4: 疼痛
    '闻到的内容': spaces.Discrete(2),  # 0: 无味觉，1: 有味觉
    '感知的温度': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),  # 0: 很冷，1: 很热
    '说话状态': spaces.Discrete(2),  # 0: 不说话，1: 说话。这里简化了说话状态为二元动作。
    '抓取状态': spaces.Discrete(2),  # 0: 无抓取物体，1: 抓取物体 。这里简化抓取运动为二元动作。真实的抓取运动十分复杂，需要更复杂的动作空间。
    '困倦状态': spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),  # 0: 不困倦，1: 困倦。这里简化困倦状态为连续动作。
    '饥饿状态': spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),  # 0: 不饥饿，1: 饥饿。这里简化饥饿状态为连续动作。
}
"""

import numpy as np
# import pygame
# from pettingzoo.utils import ParallelEnv
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_wrapper_fn

from ._mpe_utils.core import Agent, Landmark, World
from ._mpe_utils.scenario import BaseScenario
from ._mpe_utils.mpe_simple_env import SimpleEnv, make_env

# from pettingzoo.utils import wrappers
from gymnasium import spaces


class raw_env(SimpleEnv, EzPickle):
    def __init__(
            self,
            N=3,
            max_cycles=25,
            continuous_actions=False,
            render_mode=None
    ):
        EzPickle.__init__(
            self,
            N=N,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(N)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "mini_virtual_nursery_v1"

        pass  # function

    pass  # class


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):

    def make_world(self, N=3):
        world = World()
        # set any world properties first
        world.dim_c = 1
        num_agents = N
        world.num_agents = num_agents
        world.agents = [Agent() for i in range(num_agents)]
        num_landmarks = 0
        # world.agents = [Agent() for i in range(1)]

        agents_name = [f'婴儿-{i}' for i in range(1, N + 1)]

        # add agents
        for i, agent in enumerate(world.agents):
            agent.name = agents_name[i]
            agent.collide = True
            agent.silent = False
            agent.blind = False
            agent.movable = True
            agent.size = 0.050
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        return world
        pass  # function

    def reset_world(self, world, np_random):
        # random properties for agents
        agents_color = [
            np.array([1.0, 0.0, 0.0]),  # 红色
            np.array([0.0, 1.0, 0.0]),  # 绿色
            np.array([0.0, 0.0, 1.0]),  # 蓝色
        ]
        for i, agent in enumerate(world.agents):
            agent.color = agents_color[i]
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75])
        # world.landmarks[0].color = np.array([0.75, 0.25, 0.25])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.state.看到的内容 = np.zeros((640, 480, 3), dtype=np.uint8)
            # agent.state.听到的内容 = np.zeros(256, dtype=np.uint32)
            agent.state.听到的内容 = [0x10FFFF + 1] * np.ones(256, dtype=np.uint32)
            agent.state.呈现的表情 = 0
            agent.state.摸到的内容 = 0
            agent.state.闻到的内容 = 0
            agent.state.感知的温度 = 37.0
            agent.state.说话状态 = 0
            agent.state.抓取状态 = 0
            agent.state.困倦状态 = 0
            agent.state.饥饿状态 = 0
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        pass  # function

    def reward(self, agent, world):
        """
        奖励相关的内容在自行设计的模型体现。
        """
        reward = 0
        return reward
        pass  # function

    def observation(self, agent, world):

        image_data = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)  # DEBUG 这个需要用自己的模型输出表示
        # text_data = "一段测试文本" # DEBUG 这个需要用自己的模型输出表示
        text_data = [0x10FFFF + 1] * np.ones(258, dtype=np.uint32)  # DEBUG 这个需要用自己的模型输出表示

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        agent_pos = []
        for agent in world.agents:
            agent_pos.append(agent.state.p_pos - agent.state.p_pos)
        agent_color = []
        for agent in world.agents:
            agent_color.append(agent.color)
        # 展开 image_data 为一维数组
        image_data = image_data.flatten()
        agent_看到的内容 = image_data
        # for agent in world.agents:
        #     agent_看到的内容.append(agent.state.看到的内容)
        agent_听到的内容 = text_data
        # 确保 agent_听到的内容 长度为 256
        max_length = world.dim_听觉
        if len(agent_听到的内容) >= max_length:
            agent_听到的内容 = agent_听到的内容[:max_length]
        else:
            agent_听到的内容 = np.concatenate([agent_听到的内容, np.zeros(max_length - len(agent_听到的内容), dtype=np.uint32)])
        agent_呈现的表情 = 0
        agent_摸到的内容 = 0
        agent_闻到的内容 = 0
        agent_感知的温度 = 37.0
        agent_说话状态 = 0
        agent_抓取状态 = 0
        agent_困倦状态 = 0.0
        agent_饥饿状态 = 0.0

        array_observation_010 = np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + entity_color
            + agent_pos
            + agent_color
        )

        array_observation_020 = np.concatenate(
            (
                agent_看到的内容,
                agent_听到的内容,
            )
        )

        array_observation_030 = np.array([
            agent_摸到的内容,
            agent_闻到的内容,
            agent_感知的温度,
            agent_说话状态,
            agent_抓取状态,
            agent_困倦状态,
            agent_饥饿状态,
            agent_呈现的表情
        ])

        return np.concatenate([array_observation_010, array_observation_020, array_observation_030])
        pass  # function

    def encode_unicode_to_ascii_array(unicode_string: str) -> np.ndarray:
        """Encode a Unicode string to an array of ASCII integers."""
        byte_array = unicode_string.encode('utf-8')
        return np.frombuffer(byte_array, dtype=np.uint8)

    def decode_ascii_array_to_unicode(ascii_array: np.ndarray) -> str:
        """Decode an ASCII array back to a Unicode string."""
        byte_array = ascii_array.astype(np.uint8).tobytes()
        return byte_array.decode('utf-8')

    def close(self):
        pass  # function

    pass  # class
