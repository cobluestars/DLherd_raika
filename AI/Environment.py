# environment.py: 학습 환경을 정의하는 파일. 에이전트가 상호작용할 학습 환경을 시뮬레이션함.

import numpy as np
import random

class Environment:
    """강화학습 환경"""
    def __init__(self, state_size, action_size, state_change_rules, reward_rules, check_if_done):
        self.state_size = state_size # 환경의 상태 개수 (상태 vector 크기)
        self.action_size = action_size # 가능한 행동의 수
        self.state = np.zeros(state_size) # 초기 상태 (vector)
        self.state_change_rules = state_change_rules # 사용자 정의 상태 변화 규칙
        self.reward_rules = reward_rules # 사용자 정의 보상 규칙
        self.check_if_done = check_if_done # 사용자 정의 종료 조건 함수

    def reset(self):
        """환경을 초기 상태로 리셋"""
        self.state = np(self.state_size)
        return self.state
    
    def step(self, action):
        """
        행동을 취하고 다음 상태, 보상, 종료 여부를 반환
        
        :param action: 에이전트가 선택한 행동
        :return: 다음 상태, 획득한 보상, 에피소드가 종료되었는지 여부
        """
        # 사용자 정의 상태 변화 규칙에 따라 상태 변화
        if action in self.state_change_rules:
            self.state = self.state_change_rules[action](self.state)

        # 사용자 정의 보상 규칙에 따라 보상 계산
        reward = 0
        if action in self.reward_rules:
            reward = self.reward_rules[action](self.state)

        # 종료 여부 확인
        done = self.check_if_done(self.state, action, reward)

        return self.state, reward, done
    
    # Model-based 강화학습을 위한 부분
    def predict_next_state(self, state, action):
        """
        주어진 상태와 행동에 기반, 다음 상태를 예측
        
        :param state: 현재 상태
        :param action: 수행될 행동
        :return: 예측된 다음 상태
        """
        if action in self.state_change_rules:
            return self.state_change_rules[action](state)
        return state # 변화 규칙이 정의되지 않을 시 현재 상태를 반환

    # Model-based 강화학습을 위한 부분
    def predict_reward(self, state, action):
        """
        주어진 상태와 행동에 기반해 예상 보상을 계산함.

        :param state: 현재 상태
        :param action: 수행될 행동
        :return: 예측된 보상
        """
        if action in self.reward_rules:
            return self.reward_rules[action](state)
        return 0 # 보상 규칙이 정의되지 않은 경우 0 반환

    def render(self):
        """환경을 시각화함. (여기에서는 일단 상태를 출력함.)"""
        print("Current state:", self.state)


"""
def user_defined_done_condition(state, action, reward):
    # 사용자 정의 종료 조건 예시
    # 예시: 보상이 1 이상이면 학습 종료
    if reward >= 1:
        return True
    return False

#환경 생성 시 사용자 정의 종료 조건을 전달함
env = Environment(state_size=4, action_size=2, state_change_rules={}, reward_rules={}, check_if_done=user_defined_done_condition)
"""