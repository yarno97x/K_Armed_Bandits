import random

class Bandit :
    def __init__(self, stationary = True, gaussian = True):
        self.reward = random.random() - 1/2
        self.std = abs(random.random())
        self.gaussian = gaussian
        self.stationary = stationary

    def get_reward(self) :
        if self.gaussian :
            return random.gauss(self.reward, self.std)
        else :
            return random.uniform(self.reward - self.std, self.reward + self.std)
    
    def update_parameters(self) :
        if not self.stationary :
            self.reward = random.random() - 1/2
            self.std = abs(random.random())

    def __repr__(self) :
        return f"{self.reward:.1f} | {self.std:.1f}"