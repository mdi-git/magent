import numpy as np

# Carbon emission factor
CARBON_FACTOR = 0.4448

class ESSEnv:
    """
    ESS environment class (shared by A3C and genetic algorithm)
    State: [solar, wind, powermeter, ess_soc]
    Action: (action, amount) - action: 1(charge), 0(idle), -1(discharge), amount: kWh
    Reward: carbon reduction amount
    """
    def __init__(self, df, max_charge=50, init_soc=25):
        self.df = df.reset_index(drop=True)
        self.max_charge = max_charge
        self.init_soc = init_soc
        self.reset()

    def reset(self):
        self.idx = 0
        self.ess_soc = self.init_soc
        return self._get_obs()

    def _get_obs(self):
        row = self.df.iloc[self.idx]
        return np.array([
            row['solar'],
            row['wind'],
            row['powermeter'],
            self.ess_soc
        ], dtype=np.float32)

    def step(self, action, amount):
        # action: 1(charge), 0(idle), -1(discharge)
        # amount: kWh
        done = False
        reward = 0
        prev_soc = self.ess_soc
        if action == 1:  # charge
            charge = min(amount, self.max_charge - self.ess_soc)
            self.ess_soc += charge
            reward = charge * CARBON_FACTOR
        elif action == -1:  # discharge
            discharge = min(amount, self.ess_soc)
            self.ess_soc -= discharge
            reward = 0  # no carbon reduction on discharge
        # no state change for idle action (0)
        self.idx += 1
        if self.idx >= len(self.df):
            done = True
            # Prevent index overflow: return zero vector
            obs = np.zeros(4, dtype=np.float32)
            return obs, reward, done, {'ess_soc': self.ess_soc, 'prev_soc': prev_soc}
        return self._get_obs(), reward, done, {'ess_soc': self.ess_soc, 'prev_soc': prev_soc}