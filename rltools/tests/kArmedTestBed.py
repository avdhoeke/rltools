import numpy as np
from typing import Union
from rltools.tabular_solution_methods.mab import GradientBandit, SimpleBandit


def k_armed_testbed(bandit: Union[SimpleBandit, GradientBandit],
                    it: int,
                    time_steps: int,
                    **kwargs):

    # Define custom values for eps-greedy, UCB and gradient bandit algorithms
    eps = kwargs.get("eps", None)
    alpha = kwargs.get("alpha", None)
    c = kwargs.get("c", None)

    # Define initial parameters
    cls, k = bandit.__class__, bandit.k
    average_rewards = np.zeros((it, time_steps))

    for i in range(it):
        q = np.random.normal(0, 1, k)
        rewards = np.asarray(list(map(lambda x: np.random.normal(x, 1, 1000), q)))

        bandit = cls(k, rewards)
        _ = bandit.learn(time_steps=time_steps, eps=eps, c=c, alpha=alpha)

        average_rewards[i] = _

    return np.mean(average_rewards, axis=0)
