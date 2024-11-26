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
from __future__ import annotations
import logging
import os
import pygame
from gymnasium.utils import seeding
import numpy as np
from gymnasium import spaces
from typing import Any


class Scenario:
    """
    环境主要的场景类，用于定义环境的主要场景。

    Args:
        num_agents (int): 环境中的智能体数量。
        local_ratio (float): 本地奖励比例。
        max_cycles (int): 最大回合数。
        continuous_actions (bool): 是否使用连续动作。
        render_mode (str): 渲染模式。

    """
    metadata = {
        'render_modes': ["human", "rgb_array"],
        'is_parallelizable': True,
        'render_fps': 10,
        'name': "mini_virtual_nursery_v1",
    }

    def __init__(
            self,
            num_agents=3,
            local_ratio=None,
            max_cycles=25,
            continuous_actions=False,
            render_mode=None

    ):
        self.num_agents = num_agents
        self.render_mode = render_mode
        pygame.init()
        self.viewer = None
        self.width = 700
        self.height = 700
        self.screen = pygame.Surface([self.width, self.height])
        self.max_size = 1
        self.game_font = pygame.freetype.Font(
            os.path.join(os.path.dirname(__file__), "secrcode.ttf"), 24
        )

        # Set up the drawing window

        self.renderOn = False
        self._seed()

        self.max_cycles = max_cycles
        self.world = self.make_world()
        self.continuous_actions = continuous_actions
        self.local_ratio = local_ratio

        self.reset_world(self.world, self.np_random)

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }

        self._agent_selector = AgentSelector(self.agents)

        # set spaces
        self.action_spaces = dict()
        self.actions = dict()
        self.observation_spaces = dict()
        self.observations = dict()
        state_dim = 0
        for agent in self.world.agents:
            if agent.movable:
                space_dim = self.world.dim_p * 2 + 1
            elif self.continuous_actions:
                space_dim = 0
            else:
                space_dim = 1
            if not agent.silent:
                if self.continuous_actions:
                    space_dim += self.world.dim_c
                else:
                    space_dim *= self.world.dim_c
            if agent.具有视觉:
                视觉_dim = self.world.dim_视觉
            if agent.具有听觉:
                听觉_dim = self.world.dim_听觉
            if agent.具有表情:
                表情_dim = self.world.dim_表情
            if agent.具有说话能力:
                说话_dim = self.world.dim_说话
            if agent.具有抓取运动能力:
                抓取运动_dim = self.world.dim_抓取运动
            if agent.需要睡眠:
                睡眠_dim = self.world.dim_睡眠
            if agent.需要饮食:
                饮食_dim = self.world.dim_饮食

            obs_dim = len(self._get_observations(agent, self.world))
            state_dim += obs_dim
            if self.continuous_actions:  # #HACK 这个 continuous_actions 似乎没有用了
                self.action_spaces[agent.name] = spaces.Box(
                    low=0, high=1, shape=(space_dim,)
                )
            else:
                self.action_spaces[agent.name] = spaces.Dict({
                    '移动运动': spaces.Discrete(space_dim),
                    '说话': spaces.MultiDiscrete([0x10FFFF + 1] * 说话_dim),
                    # '说话': spaces.Box(low=0, high=0x10FFFF, shape=(256,), dtype=np.uint32),  # 备选方案
                    '表情': spaces.Discrete(表情_dim),
                    '抓取运动': spaces.Discrete(抓取运动_dim),
                    '睡眠': spaces.Discrete(睡眠_dim),
                    '饮食': spaces.Discrete(饮食_dim),
                })
            self.observation_spaces[agent.name] = spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(obs_dim,),
                dtype=np.float32,
            )

        self.state_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )

        self.steps = 0

        self.current_actions = [None] * self.num_agents
        pass  # function

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 1
        num_agents = self.num_agents
        world.num_agents = num_agents
        world.agents = [Agent() for i in range(num_agents)]
        num_landmarks = 0
        # world.agents = [Agent() for i in range(1)]

        agents_name = [f'婴儿-{i}' for i in range(1, self.num_agents + 1)]

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

    def observation_space(self, agent):
        return self.observation_spaces[agent]
        pass  # function

    def action_space(self, agent):
        return self.action_spaces[agent]
        pass  # function

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        pass  # function

    # def observe(self, agent):
    #     return self._get_observations(
    #         self.world.agents[self._index_map[agent]], self.world
    #     ).astype(np.float32)
    #     pass  # function

    # def observation(self, agent):
    #     visible_entities = self.world.get_visible_entities(agent)
    #     entity_pos = []
    #     for entity in visible_entities:
    #         entity_pos.append(entity.state.p_pos - agent.state.p_pos)
    #     return np.concatenate([agent.state.p_vel] + entity_pos)
    #     pass  # function

    def state(self):
        states = tuple(
            self._get_observations(
                self.world.agents[self._index_map[agent]], self.world
            ).astype(np.float32)
            for agent in self.possible_agents
        )
        return np.concatenate(states, axis=None)
        pass  # function

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed=seed)
        self.reset_world(self.world, self.np_random)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents
        pass  # function

    ## 对单个个体设置动作
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        agent.action.说话 = np.zeros(self.world.dim_说话)
        agent.action.表情 = np.zeros(self.world.dim_表情)
        agent.action.抓取运动 = np.zeros(self.world.dim_抓取运动)
        agent.action.睡眠 = np.zeros(self.world.dim_睡眠)
        agent.action.饮食 = np.zeros(self.world.dim_饮食)

        if agent.movable:
            # physical action
            agent.action.u = np.zeros(self.world.dim_p)
            if self.continuous_actions:
                # Process continuous action as in OpenAI MPE
                # Note: this ordering preserves the same movement direction as in the discrete case
                agent.action.u[0] += action[0][2] - action[0][1]
                agent.action.u[1] += action[0][4] - action[0][3]
            else:
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if agent.具有说话能力:
            agent.action.说话 = action[0]
            action = action[1:]
        if agent.具有表情:
            agent.action.表情 = action[0]
            action = action[1:]
        if agent.具有抓取运动能力:
            agent.action.抓取运动 = action[0]
            action = action[1:]
        if agent.需要睡眠:
            agent.action.睡眠 = action[0]
            action = action[1:]
        if agent.需要饮食:
            agent.action.饮食 = action[0]
            action = action[1:]

        # make sure we used all elements of action
        assert len(action) == 0
        pass  # function

    def step(self, actions):
        if (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(self.actions)
            return
        self.actions = actions
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = self.actions[cur_agent]

        if next_idx == 0:
            # self._execute_world_step()

            ## 遍历个体设置动作
            for i, agent in enumerate(self.world.agents):
                action = self.current_actions[i]
                scenario_action = []
                if agent.movable:
                    mdim = self.world.dim_p * 2 + 1
                    if self.continuous_actions:
                        scenario_action.append(action[0:mdim])
                        action = action[mdim:]
                    else:
                        scenario_action.append(action['移动运动'] % mdim)
                        action['移动运动'] //= mdim

                if agent.具有说话能力:
                    scenario_action.append(action['说话'])
                if agent.具有表情:
                    scenario_action.append(action['表情'])
                if agent.具有抓取运动能力:
                    scenario_action.append(action['抓取运动'])
                if agent.需要睡眠:
                    scenario_action.append(action['睡眠'])
                if agent.需要饮食:
                    scenario_action.append(action['饮食'])

                ## 对单个个体设置动作
                self._set_action(scenario_action, agent, self.action_spaces[agent.name])

                pass  # for

            ## 遍历各个个体获取观测
            for i, agent in enumerate(self.world.agents):
                self.observations[agent.name] = self._get_observations(agent, self.world)
                pass  # for

            self.world.step()

            global_reward = 0.0
            if self.local_ratio is not None:
                global_reward = float(self._global_reward(self.world))

            for agent in self.world.agents:
                agent_reward = float(self.reward(agent, self.world))
                if self.local_ratio is not None:
                    reward = (
                            global_reward * (1 - self.local_ratio)
                            + agent_reward * self.local_ratio
                    )
                else:
                    reward = agent_reward

                self.rewards[agent.name] = reward

            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

        return self.observations, self.reward, self.terminations, self.truncations, self.infos

        pass  # function

    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.clock = pygame.time.Clock()
            self.renderOn = True

    def render(self):
        if self.render_mode is None:
            logging.warning(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)

        self.draw()
        if self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2))
        elif self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return

    def draw(self):
        # clear screen
        self.screen.fill((255, 255, 255))

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses)))

        # update geometry and text positions
        # text_line = 0
        for e, entity in enumerate(self.world.entities):
            # geometry
            x, y = entity.state.p_pos
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            x = (
                    (x / cam_range) * self.width // 2 * 0.9
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2
            pygame.draw.circle(
                self.screen, entity.color * 200, (x, y), entity.size * 350
            )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            pygame.draw.circle(
                self.screen, (0, 0, 0), (x, y), entity.size * 350, 1
            )  # borders
            assert (
                    0 < x < self.width and 0 < y < self.height
            ), f"Coordinates {(x, y)} are out of bounds."

            # # text
            # if isinstance(entity, Agent):
            #     if entity.silent:
            #         continue
            #     if np.all(entity.state.c == 0):
            #         word = "_"
            #     elif self.continuous_actions:
            #         word = (
            #                 "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
            #         )
            #     else:
            #         word = alphabet[np.argmax(entity.state.c)]
            #
            #     message = entity.name + " sends " + word + "   "
            #     message_x_pos = self.width * 0.05
            #     message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
            #     self.game_font.render_to(
            #         self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
            #     )
            #     text_line += 1

    def reward(self, agent, world):
        """
        奖励相关的内容在自行设计的模型体现。
        """
        reward = 0
        return reward
        pass  # function

    def _global_reward(self, world):
        rew = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            rew -= min(dists)
        return rew

    def _get_observations(self, agent, world):

        image_data = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)  # DEBUG 这个需要用自己的模型输出表示
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

    def _clear_rewards(self) -> None:
        """Clears all items in .rewards."""
        for agent in self.rewards:
            self.rewards[agent] = 0

    def _accumulate_rewards(self) -> None:
        """Adds .rewards dictionary to ._cumulative_rewards dictionary.

        Typically called near the end of a step() method
        """
        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward

    def _was_dead_step(self, action) -> None:
        """Helper function that performs step() for dead agents.

        Does the following:

        1. Removes dead agent from .agents, .terminations, .truncations, .rewards, ._cumulative_rewards, and .infos
        2. Loads next agent into .agent_selection: if another agent is dead, loads that one, otherwise load next live agent
        3. Clear the rewards dict

        Examples:
            Highly recommended to use at the beginning of step as follows:

        def step(self, action):
            if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
                self._was_dead_step()
                return
            # main contents of step
        """
        if action is not None:
            raise ValueError("when an agent is dead, the only valid action is None")

        # removes dead agent
        agent = self.agent_selection
        assert (
                self.terminations[agent] or self.truncations[agent]
        ), "an agent that was not dead as attempted to be removed"
        del self.terminations[agent]
        del self.truncations[agent]
        del self.rewards[agent]
        del self._cumulative_rewards[agent]
        del self.infos[agent]
        self.agents.remove(agent)

        # finds next dead agent or loads next live agent (Stored in _skip_agent_selection)
        _deads_order = [
            agent
            for agent in self.agents
            if (self.terminations[agent] or self.truncations[agent])
        ]
        if _deads_order:
            if getattr(self, "_skip_agent_selection", None) is None:
                self._skip_agent_selection = self.agent_selection
            self.agent_selection = _deads_order[0]
        else:
            if getattr(self, "_skip_agent_selection", None) is not None:
                assert self._skip_agent_selection is not None
                self.agent_selection = self._skip_agent_selection
            self._skip_agent_selection = None
        self._clear_rewards()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    # def close(self):
    #     pass  # function

    pass  # class


# class Spaces:
#     @staticmethod
#     def Dict(**spaces):
#         return spaces
#
#     @staticmethod
#     def Box(low, high, shape, dtype):
#         return np.zeros(shape, dtype=dtype)
#
#     @staticmethod
#     def MultiDiscrete(list_n):
#         return [[x for x in range(int(n))] for n in list_n]
#
#     @staticmethod
#     def Discrete(n: int):
#         """实现一个离散空间，返回一个从0到n-1的整数序列"""
#         return [x for x in range(int(n))]


class EntityState:  # physical/external base state of all entities
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


class AgentState(
    EntityState
):  # state of agents (including communication and internal/mental state)
    def __init__(self):
        super().__init__()
        # # communication utterance
        # self.c = None
        # 看到的内容
        self.看到的内容 = spaces.Box(low=0, high=255, shape=(640, 480, 3), dtype=np.uint8)
        # 听到的内容
        # self.听到的内容 = spaces.Text(64)  # 简化地用文本信息模拟语言语音听觉，模拟来自教育者发送的认识字词句的文本信息。最大接收长度为指定的字符
        self.听到的内容 = spaces.MultiDiscrete([0x10FFFF + 1] * 256)  # 简化地用编码的文本信息模拟语言语音听觉，模拟来自教育者发送的认识字词句的文本信息。最大接收长度为指定的编码后的数组长度
        # self.听到的内容 = spaces.box(low=0, high=0x10FFFF, shape=(256,), dtype=np.uint32)  # 备选方案
        # 摸到的内容
        self.摸到的内容 = spaces.Discrete(3)  # 0: 无摸到物体，1: 摸到物体 ，2: 抓取着物体 。这里简化摸到运动为离散动作值
        # 闻到的内容
        self.闻到的内容 = spaces.Discrete(2)  # 0: 无闻到物体，1: 闻到物体 。这里简化闻到运动为离散动作值
        # 感知的温度
        self.感知的温度 = spaces.Box(low=-20.0, high=100.0, shape=(1,), dtype=np.float32)  # -20~100摄氏度
        # 说话状态
        self.说话状态 = spaces.Discrete(2)  # 0: 不说话，1: 说话。这里简化了说话状态为二元动作。
        # 抓取状态
        self.抓取状态 = spaces.Discrete(2)  # 0: 无抓取物体，1: 抓取物体 。这里简化抓取运动为二元动作。真实的抓取运动十分复杂，需要更复杂的动作空间。
        # 困倦状态
        self.困倦状态 = spaces.Box(low=0, high=1.0, shape=(1, 1), dtype=np.float32)  # 0: 不困倦，1: 困倦。这里简化困倦状态为连续动作。
        # 饥饿状态
        self.饥饿状态 = spaces.Box(low=0, high=1.0, shape=(1, 1), dtype=np.float32)  # 0: 不饥饿，1: 饥饿。这里简化饥饿状态为连续动作。
        # 呈现的表情
        self.呈现的表情 = spaces.Discrete(6),  # 0: 静，1: 喜，2: 怒，3: 哀，4: 惧，5: 思。 这里简化了输出的表情为离散动作值


class Action:  # action of the agent
    def __init__(self):
        # physical action
        # 说话
        self.说话 = None
        # 表情
        self.表情 = None
        # 抓取运动
        self.抓取运动 = None
        # 睡眠
        self.睡眠 = None
        # 饮食
        self.饮食 = None


class Entity:  # properties and state of physical world entity
    def __init__(self):
        # name
        self.name = ""
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


class Landmark(Entity):  # properties of landmark entities
    def __init__(self):
        super().__init__()


class Agent(Entity):  # properties of agent entities
    def __init__(self):
        super().__init__()
        # agents are movable by default
        self.movable = True
        # 具有视觉
        self.具有视觉 = True
        # 具有听觉
        self.具有听觉 = True
        # 具有说话能力
        self.具有说话能力 = True
        # 具有触觉
        self.具有触觉 = True
        # 具有嗅觉
        self.具有嗅觉 = True
        # 具有温度知觉
        self.具有温度知觉 = True
        # 具有疼痛知觉
        self.具有疼痛知觉 = True
        # 具有抓取运动能力
        self.具有抓取运动能力 = True
        # 需要睡眠
        self.需要睡眠 = True
        # 需要饮食
        self.需要饮食 = True
        # 具有表情
        self.具有表情 = True
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # 视野半径（视野默认是俯视角，可以穿墙） #TODO 后续考虑加入视野角度，视野是俯视角但是从个体中心向外辐射，因此视野不能够穿越非透明的障碍物。
        self.vision_radius = 10.0  # 可视半径  #TODO 暂时还没有用起来


class World:  # multi-agent world
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # position dimensionality
        self.dim_p = 2
        # 视觉 shape
        self.shape_视觉 = (640, 480, 3)
        # 视觉 dimensionality
        self.dim_视觉 = self.shape_视觉[0] * self.shape_视觉[1] * self.shape_视觉[2]
        # 听觉 dimensionality
        self.dim_听觉 = 256
        # 说话 dimensionality
        self.dim_说话 = 256
        # 触觉 dimensionality
        self.dim_触觉 = 1
        # 嗅觉 dimensionality
        self.dim_嗅觉 = 1
        # 温度知觉 dimensionality
        self.dim_温度知觉 = 1
        # 疼痛知觉 dimensionality
        self.dim_疼痛知觉 = 1
        # 抓取运动 dimensionality
        self.dim_抓取运动 = 1
        # 睡眠行为 dimensionality
        self.dim_睡眠 = 1
        # 饮食行为 dimensionality
        self.dim_饮食 = 1
        # 呈现的表情 dimensionality
        self.dim_表情 = 1
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = (
                    np.random.randn(*agent.action.u.shape) * agent.u_noise
                    if agent.u_noise
                    else 0.0
                )
                p_force[i] = agent.action.u + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if f_b is not None:
                    if p_force[b] is None:
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_pos += entity.state.p_vel * self.dt
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(
                    np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])
                )
                if speed > entity.max_speed:
                    entity.state.p_vel = (
                            entity.state.p_vel
                            / np.sqrt(
                        np.square(entity.state.p_vel[0])
                        + np.square(entity.state.p_vel[1])
                    )
                            * entity.max_speed
                    )

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = (
                np.random.randn(*agent.action.c.shape) * agent.c_noise
                if agent.c_noise
                else 0.0
            )
            agent.state.c = agent.action.c + noise

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]


class AgentSelector:
    """Outputs an agent in the given order whenever agent_select is called.

    Can reinitialize to a new order.

    Example:
        >>> agent_selector = AgentSelector(agent_order=["player1", "player2"])
        >>> agent_selector.reset()
        'player1'
        >>> agent_selector.next()
        'player2'
        >>> agent_selector.is_last()
        True
        >>> agent_selector.reinit(agent_order=["player2", "player1"])
        >>> agent_selector.next()
        'player2'
        >>> agent_selector.is_last()
        False
    """

    def __init__(self, agent_order: list[Any]):
        self.reinit(agent_order)

    def reinit(self, agent_order: list[Any]) -> None:
        """Reinitialize to a new order."""
        self.agent_order = agent_order
        self._current_agent = 0
        self.selected_agent = 0

    def reset(self) -> Any:
        """Reset to the original order."""
        self.reinit(self.agent_order)
        return self.next()

    def next(self) -> Any:
        """Get the next agent."""
        self._current_agent = (self._current_agent + 1) % len(self.agent_order)
        self.selected_agent = self.agent_order[self._current_agent - 1]
        return self.selected_agent

    def is_last(self) -> bool:
        """Check if the current agent is the last agent in the cycle."""
        return self.selected_agent == self.agent_order[-1]

    def is_first(self) -> bool:
        """Check if the current agent is the first agent in the cycle."""
        return self.selected_agent == self.agent_order[0]

    def __eq__(self, other: AgentSelector) -> bool:
        if not isinstance(other, AgentSelector):
            return NotImplemented

        return (
                self.agent_order == other.agent_order
                and self._current_agent == other._current_agent
                and self.selected_agent == other.selected_agent
        )
