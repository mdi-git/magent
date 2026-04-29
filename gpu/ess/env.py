import numpy as np

# 탄소 배출 계수
CARBON_FACTOR = 0.4448

class ESSEnv:
    """
    ESS 환경 클래스 (A3C, 유전알고리즘 공통)
    상태: [solar, wind, powermeter, ess_soc]
    행동: (action, amount) - action: 1(충전), 0(대기), -1(방전), amount: kWh
    보상: 탄소 절감량
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
        # action: 1(충전), 0(대기), -1(방전)
        # amount: kWh
        done = False
        reward = 0
        prev_soc = self.ess_soc
        if action == 1:  # 충전
            charge = min(amount, self.max_charge - self.ess_soc)
            self.ess_soc += charge
            reward = charge * CARBON_FACTOR
        elif action == -1:  # 방전
            discharge = min(amount, self.ess_soc)
            self.ess_soc -= discharge
            reward = 0  # 방전 시 탄소 절감 없음
        # 대기(0)일 때는 아무 변화 없음
        self.idx += 1
        if self.idx >= len(self.df):
            done = True
            # 인덱스 초과 방지: 0 벡터 반환
            obs = np.zeros(4, dtype=np.float32)
            return obs, reward, done, {'ess_soc': self.ess_soc, 'prev_soc': prev_soc}
        return self._get_obs(), reward, done, {'ess_soc': self.ess_soc, 'prev_soc': prev_soc} 