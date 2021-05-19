import numpy as np
from typing import Union, Optional
from .callback import Callback


class Bandit:

    def __init__(self, k: int, rewards: np.ndarray):
        self.k = k
        self.actions = np.arange(k)
        self.rewards = rewards
        self.unbiased_constant = 0

    def constant(self, step_size: float, **kwargs) -> float:
        return step_size

    def unbiased_constant(self, step_size: float, **kwargs) -> float:
        self.unbiased_constant += step_size * (1 + self.unbiased_constant)
        return step_size/self.unbiased_constant

    def get_reward(self, a: int) -> float:
        return np.random.choice(self.rewards[a])


class SimpleBandit(Bandit):

    def __init__(self, k: int, rewards: np.ndarray, **kwargs):
        super().__init__(k, rewards)
        self.n = np.zeros(k)
        self.q = kwargs.get("q", np.zeros(k))

    def incremental(self, a: int, **kwargs):
        return 1/self.n[a]

    def epsilon_greedy(self, eps: float) -> int:
        if np.random.sample() < eps:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.q)

    def ucb(self, c: float, t: float) -> int:
        zeros = np.where(self.n == 0)[0]
        if zeros.size > 0:
            return np.random.choice(zeros)
        else:
            tmp = np.asarray(list(map(lambda x: self.q[x] + c * np.sqrt(np.log(t)/self.n[x]), self.actions)))
            return np.argmax(tmp)

    def learn(self, time_steps: int,
              eps: float = None,
              c: float = None,
              alpha: float = 0.1,
              step_size: str = "incremental",
              callback: Optional[Callback] = None,
              **kwargs):

        # Store reward at every time step
        _ = np.zeros(time_steps)

        # Set step size
        try:
            f = getattr(super(), step_size)
        except:
            f = getattr(self, step_size)

        for t in range(time_steps):

            # Select action according to action selection strategy
            a = self.epsilon_greedy(eps) if eps is not None else self.ucb(c, t)
            # Update N
            self.n[a] += 1
            # Fetch reward resulting from action a
            r = self.get_reward(a)
            # Update step size
            alpha = f(a=a, step_size=alpha)
            # Update Q
            self.q[a] += alpha * (r - self.q[a])
            # Store reward earned at time step t
            _[t] = r

        return _


class GradientBandit(Bandit):

    def __init__(self, k: int, rewards: np.ndarray, **kwargs):
        super().__init__(k, rewards)
        self.h = kwargs.get("h", np.zeros(k))

    def softmax(self, a: int) -> float:
        return np.e**(self.h[a])

    def gradient_ascent(self, a_t: int, r: float, baseline: float, alpha: float, p: np.ndarray) -> None:
        for a in self.actions:
            if a == a_t:
                self.h[a] += alpha * (r - baseline) * (1 - p[a])
            else:
                self.h[a] -= alpha * (r - baseline) * p[a]

    def learn(self, time_steps: int,
              step_size: str = "constant",
              alpha: float = 0.1,
              callback: Optional[Callback] = None,
              **kwargs):

        # Store reward at every time step
        _ = np.zeros(time_steps)

        # Initialize baseline
        r_sum, baseline = 0, 0

        # Set step size
        f = getattr(super(), step_size)

        for t in range(time_steps):

            # Compute probability of being chosen based on preferences h
            denominator = np.sum([np.e ** (self.h[b]) for b in self.actions])
            p = np.asarray(list(map(lambda x: self.softmax(x), self.actions)))/denominator
            # Select action according to action selection strategy
            a = np.random.choice(self.actions, p=p)
            # Fetch reward resulting from action a
            r = self.get_reward(a)
            # Initialize baseline
            if t == 0:
                baseline = r
            # Update step size
            alpha = f(step_size=alpha)
            # Update h
            self.gradient_ascent(a, r, baseline, alpha, p)
            # Update baseline based on new reward
            baseline += (1 / (t + 1)) * (r - baseline)
            # Store reward earned at time step t
            _[t] = r

        return _


