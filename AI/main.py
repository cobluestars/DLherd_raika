# main.py: 학습을 시작하고 제어하는 주 실행 스크립트

import numpy as np
from model import QNetwork
from Environment import Environment
from agent import Agent
from keras.optimizers import Adam
from utils import plot_learning_curve #학습 과정 시각화

def main(state_size=4, action_size=2, num_episodes=1000, replay_batch_size=32, learning_rate=0.001):
    """
    학습 프로세스를 구성하고 실행하는 메인 함수

    :param state_size: 상태 벡터의 차원
    :param action_size: 가능한 액션의 수
    :param num_episodes: 총 에피소드 수
    :param replay_batch_size: 경험 리플레이 미니배치 크기
    :param learning_rate: 학습률 
    """

    #모델 초기화 및 컴파일
    model = QNetwork(state_size, action_size)
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

    # 환경 및 에이전트 인스턴스 생성
    env = Environment(state_size, action_size)
    agent = Agent(state_size, action_size)

    rewards = [] # 각 에피소드별 보상을 저장할 리스트

    for episode in range(num_episodes):
        state = env.reset() # 에피소드 시작 시 환경을 초기 상태로 리셋: 각 에피소드는 독립적 시행
        state = np.reshape(state, [1, state_size]) # 모델에 맞게 상태 형태를 조정
                                                   # [state_size]에서  [1, state_size]로 차원 변경, 배치 크기 1을 명시함

        # 이 에피소드에서 얻은 총 보상을 추적하기 위한 변수
        total_reward = 0

        # 에피소드가 끝날 때까지 (목표 달성, 실패 등) 반복
        done = False
        while not done:
            # 현재 상태에 대해 에이전트가 행동을 선택
            #이는 정책(epsilon-greedy)에 따라 결정됨
            action = agent.act(state)

            # 선택한 행동을 환경에 적용하여 다음 상태, 보상, 에피소드 종료 여부를 얻음.
            next_state, reward, done = env.step(action)

            # 다음 상태 벡터의 형태도 모델에 맞게 조정
            next_state = np.reshape(next_state, [1, state_size])

            # 에이전트의 경험을 메모리에 저장. 이 경험은 나중에 학습(겸험 리플레이)에 사용
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state # 상태 업데이트, 다음 반복에서 사용

            # 이번 반복에서 얻은 보상을 총 보상에 추가
            total_reward += reward

        #에피소드가 종료되면, 이번 에피소드의 결과 출력.
        if done:
            print(f"Episode: {episode+1}, Reward: {total_reward}")
            rewards.append(total_reward)

        # 일정 에피소드마다 주기적으로 에이전트의 경험 리플레이를 바탕으로 학습 진행
        if episode % 10 == 0:
            agent.replay(replay_batch_size)

    # 학습 과정 시각화
    plot_learning_curve(rewards, "learning_curve.png")

if __name__=='__main__':
    main()