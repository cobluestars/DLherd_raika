# main.py: 학습을 시작하고 제어하는 주 실행 스크립트

import numpy as np
from model import QNetwork
from Environment import Environment
from agent import Agent
from keras.optimizers import Adam
from utils import plot_learning_curve #학습 과정 시각화

class TrainingSession:
    def __init__(self, state_size=4, action_size=2, num_episodes=1000, replay_batch_size=32, learning_rate=0.001, evaluation_interval=50):
        """
        학습 프로세스를 구성하고 실행하는 메인 클래스

        :param state_size (int): 상태 벡터의 차원
        :param action_size (int): 가능한 액션의 수
        :param num_episodes (int): 총 에피소드 수
        :param replay_batch_size (int): 경험 리플레이 미니배치 크기
        :param learning_rate (float): 학습률
        :param evaluation_interval (int): 성능 평가를 위한 에피소드 간격
        """

        self.state_size = state_size
        self.action_size = action_size
        self.num_episodes = num_episodes
        self.replay_batch_size = replay_batch_size
        self.learning_rate = learning_rate
        self.evaluation_interval = evaluation_interval

        # 모델 초기화 및 컴파일
        self.model = QNetwork(state_size, action_size)
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

        # 환경 및 에이전트 인스턴스 생성
        self.env = Environment(state_size, action_size)
        self.agent = Agent(state_size, action_size, learning_rate=learning_rate)

        # 에이전트의 성능을 기록할 변수들 초기화
        self.epsilon_history = []
        self.learning_rate_history = [] # 학습률 변화 기록
        self.step_counts = []
        self.total_rewards = [] # 각 에피소드별 보상을 저장할 리스트

    def adjust_learning_rate(self, current_episode):
        """
        학습률을 동적으로 조정하는 함수

        - current_episode (int): 현재 에피소드
        - total_episodes (int): 총 에피소드 수
        - initial_lr (float): 초기 학습률

        Returns:
        - float: 새로운 학습률
        """
        #학습률 점진적 선형 감소 전략 구현
        decay_rate = 0.95
        new_lr = self.initial_lr * (decay_rate ** (current_episode / self.total_episodes))
        return max(new_lr, 0.0001) #학습률의 하한값 설정

    def train(self):
        for episode in range(self.num_episodes):
            state = self.env.reset() # 에피소드 시작 시 환경을 초기 상태로 리셋: 각 에피소드는 독립적 시행
            state = np.reshape(state, [1, self.state_size]) # 모델에 맞게 상태 형태를 조정
                                                            # [state_size]에서  [1, state_size]로 차원 변경, 배치 크기 1을 명시함

            # 이 에피소드에서 얻은 총 보상을 추적하기 위한 변수
            total_reward = 0

            # 에피소드별 스텝 수를 기록하기 위한 변수
            steps = 0

            # 에피소드가 끝날 때까지 (목표 달성, 실패 등) 반복
            done = False
            while not done:
                # 현재 상태에 대해 에이전트가 행동을 선택
                #이는 정책(epsilon-greedy)에 따라 결정됨
                action = self.agent.act(state)

                # 선택한 행동을 환경에 적용하여 다음 상태, 보상, 에피소드 종료 여부를 얻음.
                next_state, reward, done = self.env.step(action)

                # 다음 상태 벡터의 형태도 모델에 맞게 조정
                next_state = np.reshape(next_state, [1, self.state_size])

                # 에이전트의 경험을 메모리에 저장. 이 경험은 나중에 학습(겸험 리플레이)에 사용
                self.agent.remember(state, action, reward, next_state, done)
                
                steps += 1 # 스텝 수 증가

                state = next_state # 상태 업데이트, 다음 반복에서 사용

                # 이번 반복에서 얻은 보상을 총 보상에 추가
                total_reward += reward

            """
            에이전트의 학습 과정에서 중요 지표들(평균 보상, 에피소드당 스텝 수, ε값의 변화)을 기록
            이를 통해 학습 진행 상황을 모니터링함.
            """

            #에피소드가 종료되면, 이번 에피소드의 결과 출력.
            if done:
                print(f"Episode: {episode+1}, Reward: {total_reward} Steps: {steps}, Epsilon: {self.agent.epsilon}")
                self.epsilon_history.append(self.agent.epsilon)
                self.step_counts.append(steps)
                self.total_rewards.append(total_reward)

            # 일정 에피소드마다 주기적으로 에이전트의 경험 리플레이를 바탕으로 학습 진행
            if episode % 10 == 0:
                self.agent.replay(self.replay_batch_size)

            # 일정 간격마다 에이전트의 성능 평가 및 학습률 조정
            if episode % self.evaluation_interval == 0:
                # 환경 모델을 사용한 시뮬레이션 성능과 실제 환경 성능 평가
                self.agent.evaluate_agent()
                # 학습률 동적 조정
                new_lr = self.adjust_learning_rate(episode)
                self.agent.adjust_learning_rate(new_lr)
                self.learning_rate_history.append(new_lr)

        # 학습 과정 시각화
        plot_learning_curve(self.total_rewards, self.epsilon_history, self.step_counts, self.learning_rate_history, "learning_curve.png")

if __name__=='__main__':
    training_session = TrainingSession()
    training_session.train()