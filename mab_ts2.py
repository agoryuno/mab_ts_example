import math 
import random
from abc import ABC, abstractmethod

import numpy as np

def calc_reward(value, init_value=1000, discount_rate=0.05, T=20):
    return np.log(value/( ((1.+discount_rate)**T)*value )) 

INIT_STOCK = 1000.
NUM_ROUNDS = 20
DISCOUNT_RATE = 0.05

OUTCOMES = {
    "green" : [0.8, 0.9, 1.1, 1.1, 1.2, 1.4],
    "blue" : [0.95, 1, 1, 1, 1, 1.1],
    "red" : [0.05, 0.2, 1, 3, 3, 3]}

STRATEGIES = [
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [1./3, 1./3, 1./3],
        [0.3, 0.6, 0.1],
        [0.1, 0.8, 0.1],
        [0.3, 0.5, 0.2],
        [0.4, 0.3, 0.3],
        [0.3, 0.3, 0.4],
        [0.3, 0.1, 0.6]
    ]

assert all([math.isclose(sum(row),1) for row in STRATEGIES])

class Strategy(ABC):
    
    @abstractmethod
    def __call__(self):
        ...

class StrategyInvest(Strategy):

    def __init__(self, shares, 
                 init_stock=INIT_STOCK, 
                 num_rounds=NUM_ROUNDS,
                 discount=DISCOUNT_RATE,
                 outcomes=OUTCOMES):
        
        """
        shares - a dictionary with fund names as keys and fund shares as 
                 values
        init_stock - the starting total number of shares in all funds
        num_rounds - the number of rounds the game will be played for
        discount - the discount rate
        outcomes - a dictionary with all possible outcomes (growth rates)
                   for each fund, where fund names are keys and outcomes are
                   lists of rates. Outcome frequencies are represented by 
                   including the same outcome more than once (order is not 
                   important), e.g. [1, 1.1, 1.1, 0.9] means that probability 
                   of 1 and 0.9 is 0.25 and the probability of 1.1 is 0.5
        """
        
        self.shares = shares
        assert all(k in self.shares for k in ("green", "blue", "red"))
        
        self.portfolio = {k : init_stock*v for k,v in self.shares.items()}
        self.prices = {fund : [1] for fund in self.portfolio.keys() }
        
        self.outcomes = outcomes
        self.num_rounds = num_rounds
        
        self.__curr_round = 0
        self.discount = discount
        self.init_capital = sum([v for v in self.get_values().values()])
    
    def _choose_outcome(self, fund):
        """ Randomly chooses an outcome for a given fund """
        return random.choice(self.outcomes[fund])
    
    def _get_portfolio_return(self):
        """ 
        Returns the ratio of the discounted portfolio value 
        to initial capital value
        """
        return self._get_total_value() / \
            (self.init_capital*(1+self.discount)**self.__curr_round)
    
    def _get_total_value(self):
        """ 
        Returns the portfolio's total value given current 
        fund share prices 
        """
        return sum([value for value in self.get_values().values()])
    
    def get_values(self):
        """
        Returns a dictionary with fund names as keys and current 
        fund values as values
        """
        return {fund : self.prices[fund][-1]*stock 
                    for fund, stock in self.portfolio.items()}
    
    def value_shares(self):
        """
        Returns a dictionary with fund names as keys and current 
        shares of the corresponding funds in the total portfolio 
        value as values
        """
        values = self.get_values()
        total_value = np.sum(list(values.values()))
        return {fund : value / total_value for fund, value in values.items()}
       
    def __call__(self):
        for i in range(self.num_rounds):
            for fund in self.portfolio.keys():
                self.prices[fund].append(
                    self.prices[fund][-1]*self._choose_outcome(fund)
                )
            self.__curr_round = i
        return self._get_portfolio_return()
    
class StrategyInvestRebalancing(StrategyInvest):
    
    def __call__(self):
        ret = super().__call__()
        self.rebalance()
        return ret

    def rebalance(self):
        values = self.get_values()
        total_value = self._get_total_value()
        value_shares = self.value_shares()
        share_diffs = {fund : share - self.shares[fund] 
                        for fund, share in value_shares.items()}

        cash = 0
        for fund, diff in share_diffs.items():
            if diff >= 0.1:
                cutback = (diff - 0.1)*total_value
                cash += cutback
                self.portfolio[fund] -= cutback/self.prices[fund][-1]
        if cash > 0:
            buyins = []
            for fund, diff in share_diffs.items():
                if diff < 0:
                    buyins.append ( 
                        (fund, -1*(total_value*diff)/self.prices[fund][-1],
                                    -1*(total_value*diff), diff) 
                    )
            if len(buyins) == 1:
                buyin = buyins[0]
                quant_buy = cash/self.prices[buyin[0]][-1]
                self.portfolio[buyin[0]] += quant_buy
                return
            total_diff = np.sum([b[-1]*-1 for b in buyins])
            diff_shares = [(b[0],(b[-1]*-1)/total_diff) for b in buyins]
            for ds in diff_shares:
                quant_buy = (cash*ds[1])/self.prices[ds[0]][-1]
                self.portfolio[ds[0]] += quant_buy

class Arm:

    def __init__(self, play):
        """
        play - a callable object that returns a reward value
        """
        self._play = play
        self.rewards = []
    
    @property
    def times_played(self):
        """
        Number of times this arm's been played
        """
        return len(self.rewards)
        
    @property
    def mean_reward(self):
        """
        The mean reward received over all plays of this arm
        """
        return np.mean(self.rewards)
    
    def update(self):
        raise NotImplementedError
    
    def __call__(self):
        reward = self._play()
        self.rewards.append(reward)
        self.update()

class ArmNormalGamma(Arm):

    def __init__(self, play, alpha=1, beta=1, mu=0, lmd=0, **kwargs):
        super().__init__(play, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.lmd = lmd

    def update(self):
        n = self.times_played
        m = self.mean_reward
        self.mu = (self.lmd*self.mu + n*m)/ \
                  (self.lmd + n)
        self.lmd +=  n
        self.alpha += n/2.
        rewards = np.array(self.rewards)
        self.beta += 0.5*np.sum((rewards-m)**2) + \
                (n*self.lmd)/(self.lmd+n) * (0.5 * (m - self.mu)**2)

    def sample(self):
        tau = gamma.rvs(self.alpha, scale=1./self.beta)
        return norm.rvs(self.mu, scale=np.sqrt(1/(self.lmd*tau)) )

    def __call__(self):
        reward = self._play()
        if reward == 0:
            reward = 1e-9
        reward = np.log(reward)
        self.rewards.append(reward)
        self.update()