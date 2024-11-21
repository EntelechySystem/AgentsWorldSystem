import numpy as np
from gymnasium import spaces


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
        # communication utterance
        self.c = None
        # 看到的内容
        self.看到的内容 = spaces.Box(low=0, high=255, shape=(640, 480, 3), dtype=np.uint8)
        # 听到的内容
        # self.听到的内容 = spaces.Text(64)  # 简化地用文本信息模拟语言语音听觉，模拟来自教育者发送的认识字词句的文本信息。最大接收长度为指定的字符
        self.听到的内容 = spaces.MultiDiscrete([0x10FFFF + 1] * 256)  # 简化地用编码的文本信息模拟语言语音听觉，模拟来自教育者发送的认识字词句的文本信息。最大接收长度为指定的编码后的数组长度
        # self.听到的内容 = spaces.Box(low=0, high=0x10FFFF, shape=(256,), dtype=np.uint32)  # 备选方案
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
        self.u = None
        # communication action
        self.c = None
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
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
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
        # communication channel dimensionality
        self.dim_c = 1
        # position dimensionality
        self.dim_p = 2
        # 视觉 dimensionality
        self.dim_视觉 = 640 * 480 * 3
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
