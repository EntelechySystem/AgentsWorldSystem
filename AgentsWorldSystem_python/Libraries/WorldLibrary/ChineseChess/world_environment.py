"""
根据 Yui 的象棋棋盘及其运行的代码，适配 pettingzoo 。感谢 Yui 的分享。
"""
import tkinter as tk
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import ParallelEnv

class YuiMove:
    def __init__(self, start, end, route=None):
        if route is None:
            route = []
        self.start = start
        self.end = end
        self.route = route

def get_route(p, loc1, loc2):
    route = []
    if p[1] in ['車', '砲']:
        if loc1[0] == loc2[0]:
            xy1 = loc1[1]
            xy2 = loc2[1]
            dim = 1
        else:
            xy1 = loc1[0]
            xy2 = loc2[0]
            dim = 0
        if xy2 > xy1:
            d = 1
        else:
            d = -1
        loc = loc1
        while 1:
            if dim == 0:
                loc = (loc[0] + d, loc[1])
            else:
                loc = (loc[0], loc[1] + d)

            if loc == loc2:
                break
            route.append(loc)
    elif p[1] == '馬':
        d1 = abs(loc1[0] - loc2[0])
        if d1 == 2:
            if loc2[0] > loc1[0]:
                route.append((loc1[0] + 1, loc1[1]))
            else:
                route.append((loc1[0] - 1, loc1[1]))
        else:
            if loc2[1] > loc1[1]:
                route.append((loc1[0], loc1[1] + 1))
            else:
                route.append((loc1[0], loc1[1] - 1))
    elif p[1] == '象':
        d1 = loc2[0] - loc1[0]
        d2 = loc2[1] - loc1[1]
        route.append((loc1[0] + int(d1 / 2), loc1[1] + int(d2 / 2)))
    return route

def get_move(p, loc):
    x = loc[0]
    y = loc[1]
    r = []
    if p[1] in ['車', '砲']:
        for x2 in range(9):
            if x2 == x:
                pass
            else:
                loc2 = (x2, y)
                route = get_route(p, loc, loc2)
                m = YuiMove(loc, loc2, route)
                r.append(m)

        for y2 in range(10):
            if y2 == y:
                pass
            else:
                loc2 = (x, y2)
                route = get_route(p, loc, loc2)
                m = YuiMove(loc, loc2, route)
                r.append(m)

    elif p[1] == '馬':
        r2 = [(x + 1, y + 2), (x - 1, y + 2),  # D
              (x + 2, y + 1), (x + 2, y - 1),  # R
              (x + 1, y - 2), (x - 1, y - 2),  # U
              (x - 2, y + 1), (x - 2, y - 1)]  # L

        for loc2 in r2:
            if 0 <= loc2[0] <= 8 and 0 <= loc2[1] <= 9:
                route = get_route(p, loc, loc2)
                m = YuiMove(loc, loc2, route)
                r.append(m)

    elif p[1] == '象':
        r2 = [(x + 2, y + 2),
              (x + 2, y - 2),
              (x - 2, y + 2),
              (x - 2, y - 2)]

        for loc2 in r2:
            logi1 = (p[0] == 'r' and 0 <= loc2[0] <= 8 and 5 <= loc2[1] <= 9)
            logi2 = (p[0] == 'b' and 0 <= loc2[0] <= 8 and 0 <= loc2[1] <= 4)
            if logi1 or logi2:
                route = get_route(p, loc, loc2)
                m = YuiMove(loc, loc2, route)
                r.append(m)

    elif p[1] == '士':
        r2 = [(x + 1, y + 1),
              (x + 1, y - 1),
              (x - 1, y + 1),
              (x - 1, y - 1)]

        for loc2 in r2:
            logi1 = (p[0] == 'r' and 3 <= loc2[0] <= 5 and 7 <= loc2[1] <= 9)
            logi2 = (p[0] == 'b' and 3 <= loc2[0] <= 5 and 0 <= loc2[1] <= 2)
            if logi1 or logi2:
                m = YuiMove(loc, loc2)
                r.append(m)

    elif p[1] == '卒':
        if p[0] == 'r' and y <= 4:  # 過河
            r2 = [(x + 1, y),
                  (x - 1, y),
                  (x, y - 1)]
            for loc2 in r2:
                x2 = loc2[0]
                y2 = loc2[1]
                if 0 <= x2 <= 8 and 0 <= y2 <= 9:
                    m = YuiMove(loc, loc2)
                    r.append(m)

        elif p[0] == 'r' and y > 4:  # 沒過河
            r2 = [(x, y - 1)]
            for loc2 in r2:
                x2 = loc2[0]
                y2 = loc2[1]
                if 0 <= x2 <= 8 and 0 <= y2 <= 9:
                    m = YuiMove(loc, loc2)
                    r.append(m)

        elif p[0] == 'b' and y > 4:  # 過河
            r2 = [(x + 1, y),
                  (x - 1, y),
                  (x, y + 1)]
            for loc2 in r2:
                x2 = loc2[0]
                y2 = loc2[1]
                if 0 <= x2 <= 8 and 0 <= y2 <= 9:
                    m = YuiMove(loc, loc2)
                    r.append(m)

        elif p[0] == 'b' and y <= 4:  # 沒過河
            r2 = [(x, y + 1)]
            for loc2 in r2:
                x2 = loc2[0]
                y2 = loc2[1]
                if 0 <= x2 <= 8 and 0 <= y2 <= 9:
                    m = YuiMove(loc, loc2)
                    r.append(m)

    elif p[1] == '將':
        r2 = [(x + 1, y),
              (x - 1, y),
              (x, y + 1),
              (x, y - 1)]
        for loc2 in r2:
            x2 = loc2[0]
            y2 = loc2[1]
            logi1 = (p[0] == 'r' and 3 <= x2 <= 5 and 7 <= y2 <= 9)
            logi2 = (p[0] == 'b' and 3 <= x2 <= 5 and 0 <= y2 <= 2)
            if logi1 or logi2:
                m = YuiMove(loc, loc2)
                r.append(m)
    return r

class ChineseChessEnv(ParallelEnv):
    def __init__(self):
        self.agents = ["red", "black"]
        self.agent_selection = agent_selector(self.agents)
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.reset()

    def reset(self):
        self.space = {
            (0, 0): "b車1", (1, 0): "b馬1", (2, 0): "b象1", (3, 0): "b士1", (4, 0): "b將", (5, 0): "b士2", (6, 0): "b象2",
            (7, 0): "b馬2", (8, 0): "b車2",
            (1, 2): "b砲1", (7, 2): "b砲2",
            (0, 3): "b卒1", (2, 3): "b卒2", (4, 3): "b卒3", (6, 3): "b卒4", (8, 3): "b卒5",
            (0, 9): "r車2", (1, 9): "r馬2", (2, 9): "r象2", (3, 9): "r士2", (4, 9): "r將", (5, 9): "r士1", (6, 9): "r象1",
            (7, 9): "r馬1", (8, 9): "r車1",
            (1, 7): "r砲2", (7, 7): "r砲1",
            (0, 6): "r卒5", (2, 6): "r卒4", (4, 6): "r卒3", (6, 6): "r卒2", (8, 6): "r卒1"
        }
        self.select = (-1, -1)
        self.current_agent = "red"
        self.dones = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        return self.observe()

    def observe(self):
        return self.space

    def step(self, action):
        loc, loc2 = action
        p1 = self.space[loc]
        if p1 and p1[0] == self.current_agent[0]:
            r, _ = self.get_range(p1, loc)
            if loc2 in r:
                self.space[loc] = None
                self.space[loc2] = p1
                self.current_agent = "black" if self.current_agent == "red" else "red"
                self.dones = {agent: False for agent in self.agents}
                self.rewards = {agent: 0 for agent in self.agents}
                self.infos = {agent: {} for agent in self.agents}
                if p1[1] == '將' and loc2[1] == 0:
                    self.dones["red"] = True
                    self.rewards["black"] = 1
                elif p1[1] == '將' and loc2[1] == 9:
                    self.dones["black"] = True
                    self.rewards["red"] = 1
        return self.observe(), self.rewards, self.dones, self.infos

    def render(self):
        """
        打印环境

        Returns:

        """

        for y in range(10):
            for x in range(9):
                p = self.space.get((x, y), None)
                if p:
                    print(p, end=" ")
                else:
                    print(" . ", end=" ")
            print()
        print()

    def get_blocker(self, m):
        """
        获取阻挡者

        Args:
            m:

        Returns:

        """
        b = 0
        for loc in m.route:
            p = self.space[loc]
            if p:
                b += 1
        return b

    def is_empty(self, m):
        """
        是否为空

        Args:
            m:

        Returns:

        """
        loc2 = m.end
        if self.space[loc2]:
            return False
        else:
            return True

    def can_eat(self, p, loc2):
        """
        是否可以吃子

        Args:
            p:
            loc2:

        Returns:

        """
        p2 = self.space[loc2]
        if p2 and p[0] != p2[0]:
            return True
        else:
            return False

    def get_range(self, p, loc):
        m1 = get_move(p, loc)
        m2 = []
        a = []
        for m in m1:
            b = self.get_blocker(m)
            if b > 0:
                if b == 1 and p[1] == '砲':
                    a.append(m.end)
            else:
                m2.append(m)
                if p[1] != '砲':
                    a.append(m.end)

        m3 = []
        for m in m2:
            if self.is_empty(m):
                m3.append(m.end)

        for loc2 in a:
            if self.can_eat(p, loc2):
                m3.append(loc2)
        return m3, a

# 创建环境并运行
env = ChineseChessEnv()
env.reset()
env.render()
env.step(((0, 3), (0, 4)))
env.render()
env.step(((0, 6), (0, 5)))
env.render()