import numpy as np
import pygame
# from gymnasium.spaces import Text

from Libraries.WorldLibrary.Virtual2DMiniNurseryEnv_pettingzoo_mpe.world_environment import env, parallel_env, raw_env
from Libraries.WorldLibrary.Virtual2DMiniNurseryEnv_pettingzoo_mpe._mpe_utils.core import World

my_env = parallel_env(render_mode="human")
observations, infos = my_env.reset()
world = World()


def encode_unicode_to_array(unicode_string: str) -> np.ndarray:
    """编码 Unicode 字符串为 Unicode 整数数组。"""
    return np.array([ord(char) for char in unicode_string], dtype=np.uint32)


# def decode_array_to_unicode(unicode_array: np.ndarray) -> str:
#     """解码 Unicode 整数数组为 Unicode 字符串。"""
#     return ''.join([chr(code_point) for code_point in unicode_array])


def decode_unicode_array_to_string(unicode_array: np.ndarray) -> str:
    """解码 Unicode 整数数组为字符串。"""
    byte_array = (unicode_array % 0x10FFFF).astype(np.uint32).tobytes()
    return byte_array.decode('utf-32', errors='ignore')


def process_string(input_string, target_length=256, pad_char=' ', truncate_marker='...'):
    """
    处理字符串，截断或补全到指定长度。

    Args:
        input_string (str): 输入字符串
        target_length (int): 目标长度
        pad_char (str): 补全字符
        truncate_marker (str): 截断标记

    Returns:
        str: 处理后的字符串
        info: 处理信息
    """
    info = ""
    if len(input_string) > target_length:
        # 截断字符串并添加截断标记
        truncated_string = input_string[:target_length - len(truncate_marker)] + truncate_marker
        info = f"字符串长度超过 {target_length}，已截断。"
        return truncated_string, info
    elif len(input_string) < target_length:
        # 补全字符串
        padded_string = input_string + pad_char * (target_length - len(input_string))
        info = f"字符串长度不足 {target_length}，已补全。"
        return padded_string, info
    else:
        return input_string, info
        pass  # if
    pass  # function


while my_env.agents:

    # # 随机采样
    # actions = {agent: my_env.action_space(agent).sample() for agent in my_env.agents}
    # print("actions: ")  # #DEBUG
    # for i in actions:
    #     print(f"agent: {i}")
    #     print(f"说话: {decode_unicode_array_to_string(actions[i]['说话'])}")
    #     print(f"移动运动: {actions[i]['移动运动']}")
    #     print(f"抓取运动: {actions[i]['抓取运动']}")
    #     print(f"表情: {actions[i]['表情']}")
    #     print(f"睡眠: {actions[i]['睡眠']}")
    #     print(f"饮食: {actions[i]['饮食']}")
    #
    # observations, rewards, terminations, truncations, infos = my_env.step(actions)

    # 手动采样
    actions = {}
    for agent in my_env.agents:
        action_说话 = "你好，世界！"
        action_说话_encode = encode_unicode_to_array(action_说话)
        action = {
            # '说话': actions[agent]['说话'],
            '说话': encode_unicode_to_array(process_string(action_说话)[0]),
            # '移动运动': np.clip(1, my_env.action_space(agent)['移动运动'].low, my_env.action_space(agent)['移动运动'].high),
            '移动运动': np.int64(0),
            '抓取运动': np.int64(0),
            '表情': np.int64(0),
            '睡眠': np.int64(0),
            '饮食': np.int64(0),
        }
        actions[agent] = action
        pass  # for

    # #DEBUG
    print("actions: ")
    for i in actions:
        print(f"agent: {i}")
        print(f"说话: {decode_unicode_array_to_string(actions[i]['说话'])}")
        print(f"移动运动: {actions[i]['移动运动']}")
        print(f"抓取运动: {actions[i]['抓取运动']}")
        print(f"表情: {actions[i]['表情']}")
        print(f"睡眠: {actions[i]['睡眠']}")
        print(f"饮食: {actions[i]['饮食']}")

    observations, rewards, terminations, truncations, infos = my_env.step(actions)

    #  #FIXME 把 observations 里的 Numpy 数组内容转换成人类可阅读的内容
    # #DEBUG
    print("observations: ")
    for agent_name, agent_observations in observations.items():
        print(f"{agent_name}:")
        # 逆向 observations 里的 Numpy 数组内容为一个个具体的观察值
        observation_看到的内容 = agent_observations[:world.dim_视觉]
        agent_observations = agent_observations[world.dim_视觉:]
        observation_听到的内容 = agent_observations[:world.dim_听觉]
        agent_observations = agent_observations[world.dim_听觉:]
        observation_摸到的内容 = agent_observations[:world.dim_触觉]
        agent_observations = agent_observations[world.dim_触觉:]
        observation_闻到的内容 = agent_observations[:world.dim_嗅觉]
        agent_observations = agent_observations[world.dim_嗅觉:]
        observation_感知的温度 = agent_observations[0]
        agent_observations = agent_observations[1:]
        observation_说话状态 = agent_observations[0]
        agent_observations = agent_observations[1:]
        observation_抓取状态 = agent_observations[0]
        agent_observations = agent_observations[1:]
        observation_困倦状态 = agent_observations[0]
        agent_observations = agent_observations[1:]
        observation_饥饿状态 = agent_observations[0]
        agent_observations = agent_observations[1:]
        observation_呈现的表情 = agent_observations[0]
        agent_observations = agent_observations[1:]
        print(f"看到的内容: {observation_看到的内容}")
        print(f"听到的内容: {observation_听到的内容}")
        print(f"摸到的内容: {observation_摸到的内容}")
        print(f"闻到的内容: {observation_闻到的内容}")
        print(f"感知的温度: {observation_感知的温度}")
        print(f"说话状态: {observation_说话状态}")
        print(f"抓取状态: {observation_抓取状态}")
        print(f"困倦状态: {observation_困倦状态}")
        print(f"饥饿状态: {observation_饥饿状态}")
        print(f"呈现的表情: {observation_呈现的表情}")
        pass  # for

    event = pygame.event.poll()
    if event.type == pygame.QUIT:
        pygame.quit()

    pass  # while

print("运行结束！")
