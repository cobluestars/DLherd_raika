# agent.py: (환경과 모델이 준비된 후) 강화학습 에이전트와 관련된 알고리즘을 포함하는 파일

import numpy as np
from model import QNetwork
from keras.optimizers import Adam

class Agent: 
    """Q-Network를 사용하여 각 상태에서 최적의 행동을 결정하는 방법 학습하는 에이전트"""
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = [] #경험 리플레이 메모리
        self.gamma = gamma  # 할인율: 미래 보상을 현재 가치로 환산할 때 사용되는 계수
                            # 값이 0에 가까우면 에이전트는 즉각적 보상을 더 중요시하고
                            # 값이 1에 가까우면 먼 미래의 보상도 현재의 보상만큼 중요시함.
                            # 즉, 할인율은 에이전트가 얼마나 장기적인 결과를 고려할 것인지를 결정
        self.epsilon = epsilon # 탐험율: 에이전트가 무작위로 행동을 선택할 확률
                               # 에이전트가 환경에 대해 학습하기 전에 다양한 상황을 경험하게 한다.
                               # 탐험율이 높으면 에이전트는 더 많이 탐험하고, 낮으면 학습된 정책에 따라 행동한다.
        self.epsilon_min = epsilon_min # 탐험율의 하한값. 에이전트가 항상 일정 수준 이상으로
                                       # 새로운 행동을 탐험할 수 있도록 보장
        self.epsilon_decay = epsilon_decay # 각 에피소드 이후 탐험율을 감소시키는 비율
                                           # 너무 크면 에이전트의 충분한 탐험 불가능
                                           # 너무 작으면 학습이 비효율적
        self.model = QNetwork(state_size, action_size)
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

    def remember(self, state, action, reward, next_state, done):
        """에이전트의 경험을 메모리에 저장, 경험은 (상태, 행동, 보상, 다음 상태, 종료 여부)의 튜플로 저장됨"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """ε -greedy 전략을 사용하여 에이전트의 행동 결정, 무작위로 탐험하거나 학습된 정책을 활용"""
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size) # 탐험
        qs = self.model.predict(np.array([state])) # 현재 상태에 대한 행동 가치 함수 예측
        return np.argmax(qs[0]) # 가장 높은 Q값을 가진 행동 선택
    
    def replay(self, batch_size):
        """경험 리플레이를 통한 모델 학습 (저장된 경험을 사용해서 네트워크를 학습하는 역할), 미니배치를 무작위로 선택하여 학습"""
        # 메모리에 충분한 경험이 쌓이지 않았다면 함수를 종료
        if len(self.memory) < batch_size:
            return
        # 경험 메모리에서 무작위로 미니배치를 선택
        minibatch = np.random.choice(self.memory, batch_size, replace=False)
        for state, action, reward, next_state, done in minibatch:
            # 종료 상태가 아니라면, target은 현재 보상 + 할인된 최대 미래 보상
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            # 현재 상태에 대한 모델의 예측 값을 가져옴
            target_f = self.model.predict(np.array[state])
            #선택된 행동에 대한 타겟 값을 업데이트
            target_f[0][action] = target
            # 모델 업데이트
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        # 탐험율 감소
        self.update_epsilon()

    def update_epsilon(self):
        """탐험률 감소. 학습이 진행됨에 따라 탐험보다는 학습된 정책을 활용하도록 유도함."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """모델 불러오기"""
        self.model.load_weights(name)

    def save(self, name):
        """모델 저장하기"""
        self.model.save_weight(name)