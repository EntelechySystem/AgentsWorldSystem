import pygame

from Libraries.WorldLibrary.虚拟2D迷你育儿室_pettingzoo_mpe.world_environment import env, parallel_env, raw_env

my_env = parallel_env(render_mode="human")
observations, infos = my_env.reset()

while my_env.agents:
    # this is where you would insert your policy
    actions = {agent: my_env.action_space(agent).sample() for agent in my_env.agents}

    observations, rewards, terminations, truncations, infos = my_env.step(actions)

    event = pygame.event.poll()
    if event.type == pygame.QUIT:
        pygame.quit()