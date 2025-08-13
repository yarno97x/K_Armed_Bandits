from bandit import Bandit
import numpy as np
import random
import matplotlib.pyplot as plt
import math

class K_Armed_Bandit :
    def __init__(self, n = 10, stationary = True, gaussian = True, iters=1000):
        self.bandits = [Bandit(stationary=stationary, gaussian=gaussian) for _ in range(n)]
        self.stationary = stationary
        self.gain = np.zeros(iters)
        self.iters = iters
        self.n = n
        self.N = np.zeros(self.n)

    def choose_greedily(self, arr) :
        # print("Exploiting")
        return np.argmax(arr)
    
    def choose_randomly(self, arr) :
        # print("Exploring")
        return random.randint(0, len(arr) - 1)

class ValuedBandit(K_Armed_Bandit) :
    def __init__(self, n=10, stationary=True, gaussian=True, iters=1000, optimistic=True):
        super().__init__(n, stationary, gaussian, iters)
        bias = 10 if optimistic else 0
        self.value_function = np.zeros(n) + bias
        self.alpha = 1 if stationary else 0.5 

    def simulate(self) :
        for t in range(self.iters) :
            action = self.choose_action(t)
            self.N[action] += 1

            reward = self.bandits[action].get_reward()
            self.gain[t] = self.gain[t-1] + reward if t > 0 else reward

            self.update_value_function(action, reward)
        return self.gain, self.N

    def update_value_function(self, action, reward) :
        self.value_function[action] = self.value_function[action] + self.alpha * (reward  - self.value_function[action]) / self.N[action] 

    
class EPS_GREEDY(ValuedBandit) :
    def __init__(self, n=10, stationary=True, gaussian=True, eps=0.3, iters=1000, optimistic=True):
        super().__init__(n, stationary, gaussian, iters, optimistic)
        self.eps = eps

    def choose_action(self, t) :
        if random.random() < self.eps :
            return self.choose_randomly(self.value_function)
        else :
            return self.choose_greedily(self.value_function)
        

class UCB(ValuedBandit) :
    def __init__(self, n=10, stationary=True, gaussian=True, iters=1000, optimistic=True, c=1):
        super().__init__(n, stationary, gaussian, iters, optimistic)
        self.c = c

    def choose_action(self, t) :
        return self.choose_greedily([self.value_function[action] + self.c * ((math.log(t+1) / (self.N[action] + 1)) ** (1/2)) for action in range(self.n)])

class GradientBandit(K_Armed_Bandit) :
    def __init__(self, n=10, stationary=True, gaussian=True, iters=1000, alpha = 1):
        super().__init__(n, stationary, gaussian, iters)
        self.alpha = alpha
        self.preferences = np.zeros(n)
        self.baselines = np.zeros(n)

    def softmax(self, i) :
        return self.preferences[i] / sum(np.exp(self.preferences))

    def choose_action(self) :
        return np.argmax(self.preferences / sum(np.exp(self.preferences)))
    
    def update_preferences(self, action, reward) :
        for i in range(self.n) :
            if i == action :
                self.preferences[i] = self.preferences[i] + self.alpha * (reward - self.baselines[i]) * (1 - self.softmax(i)) 
            else :    
                self.preferences[i] = self.preferences[i] - self.alpha * (reward - self.baselines[i]) * self.softmax(i)
            
    def simulate(self) :
        for t in range(self.iters) :
            print(self.preferences)
            action = self.choose_action()
            self.N[action] += 1

            reward = self.bandits[action].get_reward()
            self.gain[t] = self.gain[t-1] + reward if t > 0 else reward
            self.baselines[action] = self.baselines[action] + (reward - self.baselines[action]) / (self.N[action] + 1)
            self.update_preferences(action, reward)
        return self.gain, self.N

if __name__ == "__main__" :
    kAB = GradientBandit()
    print(kAB.bandits)
    gain, N = kAB.simulate()
    plt.plot(np.arange(kAB.iters), gain)
    plt.show()
    plt.scatter(np.arange(kAB.n), N)
    plt.show()

