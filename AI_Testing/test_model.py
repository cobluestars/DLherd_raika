# test_script.py

import json
import os
from agent import Agent
from Environment import Environment
from UserDefinedItem import UserDefinedItem
from UserDefinedItems import context_based_salary_for_student, context_based_salary_for_developer, context_based_salary_for_accountant

# UserDefinedItem 인스턴스 생성을 위한 도우미 함수
def create_user_defined_items():
    # UserDefinedItems.py에서 정의한 항목들을 인스턴스화합니다.
    return [
        UserDefinedItem(
            name='TimeStamp',
            item_type='time',
            options=['2024-04-01T00:00:00', '2024-04-01T08:00:00'],
            peak_times=[
            ['2024-04-01T00:00:00', '2024-04-01T01:00:00', 0.3],
            # ['2024-04-01T03:00:00', '2024-04-01T04:00:00', 0.2],
            ['2024-04-01T07:00:00', '2024-04-01T08:00:00', 0.5]
            ]
        ),
        UserDefinedItem(
            name='job',
            item_type='array',
            options=[
                UserDefinedItem(
                    name='student',
                    item_type='array',
                    options=[
                        UserDefinedItem(name='age', item_type='number', options=[10, 30]),
                        UserDefinedItem(name='salary', item_type='number', options=[8000, 20000],
                                        contextBasedOptions=context_based_salary_for_student)
                    ]
                ),
                UserDefinedItem(
                    name='developer',
                    item_type='array',
                    options=[
                        UserDefinedItem(name='age', item_type='number', distribution='normal', mean=40, std_dev=6, options=[20, 60]),
                        UserDefinedItem(name='salary', item_type='number',
                                        contextBasedOptions=context_based_salary_for_developer)
                    ]
                ),
                UserDefinedItem(
                    name='accountant',
                    item_type='array',
                    options=[
                        UserDefinedItem(name='age', item_type='number', distribution='normal', mean=40, std_dev=6, options=[20, 60]),
                        UserDefinedItem(name='salary', item_type='number',
                                        contextBasedOptions=context_based_salary_for_accountant)
                    ]
                )
            ],
            randomizeArrays=True,
            selectionProbability=True,
            probability_settings=[
                {"identifier": "developer", "probability": 45},  # 45% 확률로 developer 선택
                {"identifier": "accountant", "probability": 45}  # 45% 확률로 accountant 선택
            ]
        ),
        UserDefinedItem(
            name='favorite drinks',
            item_type='array',
            options=['Americano', 'Energy Drink', 'Coke', 'Tea'],
            randomizeArrays=True
        ),
        UserDefinedItem(
            name='hobbies',
            item_type='object',
            options={'hobby1': 'reading', 'hobby2': 'gaming', 'hobby3': 'coding', 'hobby4': 'hiking'},
            randomizeObjects=True,
            objectSelectionCount=3,
            randomizeSelectionCount=True
        )
    ]

# 학습 환경 및 에이전트 설정
def setup_environment_and_agent():
    user_defined_items = create_user_defined_items()
    state_size = len(user_defined_items)  # 상태의 크기는 UserDefinedItem의 수에 해당합니다.
    action_size = 2  # 임의로 설정한 행동의 수

    env = Environment(user_defined_items=user_defined_items, state_size=state_size, action_size=action_size)
    agent = Agent(state_size=state_size, action_size=action_size, user_defined_items=user_defined_items)

    return env, agent

# 테스트 스크립트의 메인 실행 부분
def main():
    # 환경과 에이전트 설정
    env, agent = setup_environment_and_agent()

    # 각 에피소드에 대해 학습을 수행합니다.
    for episode in range(10):  # 10 에피소드에 대해 학습을 시뮬레이션합니다. 실제로는 더 많은 에피소드가 필요합니다.
        # 에피소드 시작 시 환경을 리셋합니다.
        state = env.reset()

        total_reward = 0
        done = False

        while not done:
            # 에이전트는 현재 상태를 기반으로 행동을 결정합니다.
            action = agent.act(state)

            # 에이전트의 행동을 환경에 적용하고 새로운 상태와 보상, 그리고 에피소드 종료 여부를 얻습니다.
            next_state, reward, done = env.step(action)

            # 에이전트는 행동의 결과를 기억합니다.
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        # 경험 리플레이를 통해 에이전트를 학습시킵니다.
        agent.replay(batch_size=32)

        print(f"에피소드 {episode + 1}: 총 보상 = {total_reward}")

    # 학습이 끝난 후, 학습된 정책을 저장합니다.
    agent.save("./saved_model")

if __name__ == "__main__":
    main()