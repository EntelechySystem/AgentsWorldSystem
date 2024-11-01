"""
烧水倒水测试环境
"""

import pettingzoo
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import ParallelEnv


# noinspection NonAsciiCharacters
class 滤水机:
    def __init__(self):
        self.转换开关 = "未过滤"
        self.水量开关 = "关闭"

    def 设置转换开关(self, 模式):
        self.转换开关 = 模式

    def 设置水量开关(self, 状态):
        self.水量开关 = 状态

    pass  # class


class 烧水壶:
    def __init__(self):
        self.水量 = 0
        self.是否烧开 = False

    def 加水(self, 量):
        self.水量 = 量
        self.是否烧开 = False

    def 烧水(self):
        self.是否烧开 = True

    pass  # class


class 容器:
    def __init__(self, 容量):
        self.容量 = 容量
        self.当前水量 = 0

    def 加水(self, 量):
        self.当前水量 = min(self.容量, self.当前水量 + 量)

    def 是否满(self):
        return self.当前水量 >= self.容量

    def 是否空(self):
        return self.当前水量 == 0

    pass  # class


class 水处理过程:
    def __init__(self):
        self.滤水机 = 滤水机()
        self.烧水壶 = 烧水壶()
        self.玻璃水瓶 = 容器(1.5)
        self.保温瓶 = 容器(1.0)
        self.水桶 = 容器(10)
        self.接水盆 = 容器(5)

    def 检查状态(self):
        return {
            "烧水壶": {"水量": self.烧水壶.水量, "是否烧开": self.烧水壶.是否烧开},
            "玻璃水瓶": self.玻璃水瓶.当前水量,
            "保温瓶": self.保温瓶.当前水量
        }

    def 取水(self, 季节):
        if 季节 == "夏天":
            self.滤水机.设置转换开关("未过滤")
            for _ in range(2):
                self.接水盆.加水(5)
                self.水桶.加水(self.接水盆.当前水量)
                self.接水盆.当前水量 = 0
        else:
            self.滤水机.设置转换开关("未过滤")
            self.接水盆.加水(0.2)
            self.水桶.加水(self.接水盆.当前水量)
            self.接水盆.当前水量 = 0

        self.滤水机.设置转换开关("已过滤")
        self.滤水机.设置水量开关("打开")
        self.烧水壶.加水(self.接水盆.容量)
        self.滤水机.设置水量开关("关闭")

    def 烧水(self):
        self.烧水壶.烧水()

    def 转换水到各容器(self):
        if not self.保温瓶.是否满() or not self.玻璃水瓶.是否满():
            if self.烧水壶.是否烧开:
                if self.烧水壶.水量 > 0:
                    if not self.保温瓶.是否满():
                        self.倒水(self.烧水壶, self.保温瓶)
                    if not self.玻璃水瓶.是否满():
                        self.倒水(self.烧水壶, self.玻璃水瓶)
        elif self.保温瓶.是否满() and not self.玻璃水瓶.是否满():
            if self.烧水壶.是否烧开:
                if self.烧水壶.水量 > 0:
                    self.倒水(self.烧水壶, self.玻璃水瓶)
        elif not self.保温瓶.是否满() and self.玻璃水瓶.是否满():
            if self.烧水壶.是否烧开:
                if self.烧水壶.水量 > 0:
                    self.倒水(self.烧水壶, self.保温瓶)

    def 倒水(self, 源容器, 标容器):
        while not 标容器.是否满() and not 源容器.是否空():
            标容器.加水(1)
            源容器.水量 -= 1

    def 主流程(self):
        while True:
            状态 = self.检查状态()
            if 状态["烧水壶"]["水量"] == 0:
                self.取水("夏天")  # 或 "冬天"
                self.烧水()
            elif 状态["烧水壶"]["水量"] > 0 and not 状态["烧水壶"]["是否烧开"]:
                self.烧水()
            elif 状态["烧水壶"]["水量"] > 0 and 状态["烧水壶"]["是否烧开"]:
                self.转换水到各容器()

    pass  # class


# 使用PettingZoo库来模拟这个过程
class 水处理环境(ParallelEnv):
    def __init__(self):
        self.过程 = 水处理过程()
        self.agents = ["agent_1"]
        self.agent_selection = agent_selector(self.agents)
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def reset(self):
        self.过程 = 水处理过程()
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        return self.过程.检查状态()

    def step(self, action):
        if action == "取水":
            self.过程.取水("夏天")
        elif action == "烧水":
            self.过程.烧水()
        elif action == "转换水到各容器":
            self.过程.转换水到各容器()
        self.dones = {agent: True for agent in self.agents}
        return self.过程.检查状态(), self.rewards, self.dones, self.infos

    def render(self):
        状态 = self.过程.检查状态()
        print(f"烧水壶: {状态['烧水壶']}, 玻璃水瓶: {状态['玻璃水瓶']}, 保温瓶: {状态['保温瓶']}")

    pass  # class


def main():
    # 创建环境并运行
    env = 水处理环境()
    env.reset()
    env.step("取水")
    env.render()
    env.step("烧水")
    env.render()
    env.step("转换水到各容器")
    env.render()


if __name__ == "__main__":
    main()
