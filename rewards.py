### - Any Custom Built Reward Functions Go Here - ###

def reward_1(obs, action):
    raise NotImplementedError

def get_rew(input: str):
    if input == 'base':
        return reward_1
    else:
        raise ValueError