# agent.py: (환경과 모델이 준비된 후) 강화학습 에이전트와 관련된 알고리즘을 포함하는 파일

"""
배치(batch): 모델 학습 시에 한 번에 네트워크에 전달되는 데이터의 집합.
             각 배치는 여러 개의 샘플(데이터 포인트)로 구성되며,
             각 샘플은 특정 작업에 필요한 정보를 담고 있음.
             강화학습의 맥락에서, 이 정보는 에이전트의 상태(state), 취한 행동(action), 받은 보상(reward), 그리고 새로운 상태(next state)를 포함.

             강화학습에서 "배치"는 주로 '경험 리플레이' 메커니즘에서 사용.
             에이전트가 환경과 상호작용하며 얻은 경험(상태, 행동, 보상, 다음 상태)을 메모리에 저장하고,
             이 메모리에서 무작위로 선택한 여러 경험을 배치로 묶어 모델을 학습시킴.
             이 과정에서 배치는 에이전트가 다양한 상황에서의 경험을 동시에 고려하며 학습하도록 도움.
"""

import math
import numpy as np
import random
import sys

# Python 모듈 검색 경로에 디렉토리 추가
sys.path.append('c:\\DLherd_raika')

from Environment import Environment
from model import QNetwork, EnvironmentModel
from UserDefinedItem import UserDefinedItem
from keras.optimizers import Adam
from keras import backend as K # K: keras의 백엔드 함수를 지칭. 여기서는 특정 모델의 옵티마이저에 설정된 학습률 값을 변경하는 데에 사용

def apply_action_to_state(current_state, action, user_defined_items):
    """
    주어진 상태에 행동을 적용하여 새로운 상태를 생성
    :param current_state: 현재 상태
    :param action: 적용할 행동, *** {'action_type': ..., 'item_name': ..., 'action_value': ...} 형태의 딕셔너리
    :param user_defined_items: UserDefinedItem 인스턴스의 리스트
    :return: 새로운 상태
    """
    new_state = current_state.copy() # 현재 상태를 복사하여 시작

    # Action을 처리하기 위한 상태 변화 로직
    # 행동의 타입에 따라 적절한 처리를 수행함.
    for item in user_defined_items:
        if item.name == action['item_name']:

            if action['action_type'] == 'option' or action['action_type'] == 'context_option':
                # 옵션 또는 조건부 옵션 기반 행동 처리
                new_state[action['item_name']] = action['action_value']

            elif action['action_type'] == 'peak_times_setting':
                # 시간 설정 기반 행동 처리
                if 'peak_times' in action:
                    item.peak_times = action['peak_times']
                new_state[item.name] = item.generate_value()

            elif action['action_type'] == 'probability_setting':
                # 확률 설정 기반 행동 처리
                probabilities = item.setting_probabilities()
                selected_option = item.apply_probabiliity_based_selection(probabilities)
                new_state[item.name] = selected_option

            elif action['action_type'] == 'distribution_adjustment':
                # 분포 조정 기반 행동 처리
                if 'distribution' in action:
                    item.distribution = action['distribution']
                if 'mean' in action:
                    item.mean = action['mean']
                if 'std_dev' in action:
                    item.std_dev = action['std_dev']
                new_state[item.name] = item.generate_value()

            elif action['action_type'] == 'randomize_option':
                # 랜덤화 옵션 기반 행동 처리
                if 'randomizeArrays' in action:
                    item.randomizeArrays = action['randomizeArrays']
                if 'arraySelectionCount' in action:
                    item.arraySelectionCount = action['arraySelectionCount']
                if 'randomizeObjects' in action:
                    item.randomizeObjects = action['randomizeObjects']
                if 'objectSelectionCount' in action:
                    item.objectSelectionCount = action['objectSelectionCount']
                new_state[item.name] = item.generate_value()

            # 통계적 지표에 대한 조정 행동 처리
            elif action['action_type'] == 'adjust_mode':
                item.mode = action['new_value'] # mode 값 조정
            elif action['action_type'] == 'adjust_median':
                item.median = action['new_value'] # median 값 조정
            elif action['action_type'] == 'adjust_weighted_mean':
                item.weighted_mean = action['new_value'] # weighted_mean 값 조정
            elif action['action_type'] == 'adjust_geometric_mean':
                item.geometric_mean = action['new_value'] # geometric_mean 값 조정 

    return new_state


class MCTSNode:
    def __init__(self, state, parent=None, user_defined_items=None, transition_function=None, reward_function=None, termination_condition=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.user_defined_items = user_defined_items or [] # UserDefinedItem들의 집합 (리스트)
        self.untried_actions = self.get_possible_actions()

        #simulation 메소드 내부의 보상 함수, 종료 조건 함수 커스텀 가능
        self.reward_function = reward_function
        self.termination_condition = termination_condition        

    def get_possible_actions(self):
        # 현재 상태에서 가능한 모든 행동을 반환
        possible_actions = []
        for item in self.user_defined_items:
            # 기본 옵션을 행동으로 추가
            if item.options:
                for option in item.options:
                    possible_actions.append({'action_type': 'option', 'item_name': item.name, 'action_value': option})

            # 조건부 옵션을 평가하여 행동으로 추가함
            if item.contextBasedOptions:
                context_based_options = item.evaluate_context_based_options(self.state)
                if context_based_options and 'options' in context_based_options:
                    for option in context_based_options['options']:
                        possible_actions.append({'action_type': 'context_option', 'item_name': item.name, 'action_value': option})

            # 시간 설정을 기반으로 행동 추가
            if item.peak_times:
                action = {'action_type': 'peak_times', 'item_name': item.name, 'peak_times': item.peak_times}
                possible_actions.append(action)

            # 확률 설정을 기반으로 행동 추가
            if item.probability_settings:
                for setting in item.probability_settings:
                    possible_actions.append({'action_type': 'probability_setting', 'item_name': item.name, 'action_value': setting})

            # 분포, 수치 조정 등의 세부 사항들을 행동으로 추가
            if item.distribution or item.mean or item.std_dev:
                action = {'action_type': 'distribution_adjustment', 'item_name': item.name, 'distribution': item.distribution, 'mean': item.mean, 'std_dev': item.std_dev}
                possible_actions.append(action)

            # 배열, 객체 랜덤화 옵션 등을 행동으로 추가
            if item.randomizeArrays or item.randomizeObjects:
                possible_actions.append({'action_type': 'randomize_option', 'item_name': item.name, 'randomizeArrays': item.randomizeArrays, 'arraySelectionCount': item.arraySelectionCount, 'randomizeObjects': item.randomizeObjects, 'objectSelectionCount': item.objectSelectionCount})

            # 통계적 지표에 대한 조정을 행동으로 추가
            if item.mode is not None:
                possible_actions.append({'action_type': 'adjust_mode', 'item_name': item.name, 'new_value': item.mode}) # 새로운 mode 값 설정
            if item.median is not None:
                possible_actions.append({'action_type': 'adjust_median', 'item_name': item.name, 'new_value': item.median}) # 새로운 median 값 설정
            if item.weighted_mean is not None:
                possible_actions.append({'action_type': 'adjust_weighted_mean', 'item_name': item.name, 'new_value': item.weighted_mean}) # 새로운 weighhted_mean 값 설정
            if item.geometric_mean is not None:
                possible_actions.append({'action_type': 'adjust_geometric_mean', 'item_name': item.name, 'new_value': item.geometric_mean}) # 새로운 geometric_mean 값 설정

        return possible_actions

    def select_child(self):
        # UCB1(Upper Confidence Bound) 공식을 사용해서 최적의 자식 노드를 선택
        # 메소드는 탐색(Exploration)과 활용(Exploitation) 사이에 균형을 맞추는 데에 도움을 줌
        """
        UCB1 공식: win_rate + sqrt(2 * log(parent_visits) / child_visits)
        - win_rate: 자식 노드의 승률 (노드의 승리 횟수 / 노드의 방문 횟수)
        - parent_visits: 현재 (부모) 노드의 총 방문 횟수
        - child_visits: 자식 노드의 방문 횟수

        이 공식은 자식 노드의 승률(활용)과 해당 자식 노드를 탐색한 횟수(탐색)를 모두 고려함
        탐색 파라미터는 탐색의 정도를 조절하며, 일반적으로 sqrt(2)가 사용됨

        UCB1 공식은 자식 노드를 선택할 때, 그 노드의 성공률(활용)과 그 노드가 얼마나 적게 탐색되었는지(탐색)를
        모두 고려하여 최적의 균형을 찾는 데에 도움을 줌. 이 공식을 통해,
        MCTS는 너무 자주 방문되는 노드만을 선택하는 걸 피하고, 덜 탐색된 노드에 대한 탐색 기회를 제공하여
        전체 탐색 공간을 효과적으로 탐색할 수 있음.
        """

        # 선택된 자식 노드가 없으면 None 반환
        if not self.children:
            return None
        
        best_score = float('-inf') #최고 점수를 저장할 변수, 디폴트 값은 무한대 음수
        best_child = None # 최적의 자식 노드를 저장할 변수

        for child in self.children:
            # 자식 노드의 평균 승률 계산
            win_rate = child.wins / child.visits if child.visits > 0 else 0

            # UCB1 공식을 적용해 자식 노드의 점수 계산
            # math.sqrt((2 * math.log(self.visits)) / child.visits): 탐색을 장려하는 항목
            ucb1 = win_rate + math.sqrt((2 * math.log(self.visits)) / child.visits)

            # 가장 높은 UCB1 점수를 가진 자식 노드를 선택
            if ucb1 > best_score:
                best_score = ucb1
                best_child = child

        return best_child

    def expand(self):
        # 미시도된 행동 중 하나를 선택, 해당 행동을 취한 결과로 새로운 자식 노드를 생성
        # 해당 메소드는 탐색 과정에서 새로운 가능성을 열어주는 역할

        """
        과정:
        1. 현재 노드에서 가능한 행동 중 아직 시도하지 않은 행동을 선택
        2. 선택된 행동을 현재 상태에 적용하여 새로운 상태를 생성함.
        3. 새로운 상태를 기반으로 새로운 자식 노드를 생성하고 현재 노드의 자식으로 추가함.
        4. 생성된 새 자식 노드를 반환함.
        """

        # 아직 시도하지 않은 행동을 선택함
        if not self.untried_actions:
            return None # 모든 행동을 시도했다면 더는 확장할 수 없음
        
        # 시도하지 않은 행동 중 하나를 무작위로 선택
        action = random.choice(self.untried_actions)
        self.untried_actions.remove(action)

        # 선택된 행동을 현재 상태에 적용하여 새로운 상태를 생성
        new_state = apply_action_to_state(self.state, action, self.user_defined_items)

        # 새로운 자식 노드 생성
        new_child = MCTSNode(state=new_state, parent=self, user_defined_items=self.user_defined_items)

        # 새로운 자식 노드를 현재 노드의 자식 목록에 추가
        self.children.append(new_child)

        return new_child

    def simulate(self):
        # 현재 노드에서 게임 종료 상태까지 시뮬레이션 수행. 결과를 바탕으로 보상을 계산하여 반환.
        current_state = self.state # 시작 상태 설정
        total_reward = 0 # 총 보상 초기화
        done = False # 게임 종료 플래그 초기화

        while not done:
            possible_actions = self.get_possible_actions() # 가능한 모든 행동 가져오기
            action = random.choice(possible_actions) if possible_actions else None # 가능한 행동 중 하나를 무작위로 선택
            if not action:
                break # 가능한 행동이 없으면 반복을 종료

            # 상태 전이: 상태 전이 함수(apply_action_to_state) 사용하여 상태 변화 처리
            new_state = apply_action_to_state(current_state, action, self.user_defined_items)

            # 보상 계산: 사용자 정의 보상 함수가 있다면 해당 함수 사용, 아니면 무작위 보상
            if self.reward_function:
                reward = self.reward_function(current_state, action, new_state)
            else:
                reward = random.uniform(-1, 1) # 기본 무작위 보상

            total_reward += reward # 총 보상에 현재 단계의 보상 추가
            current_state = new_state # 상태 업데이트

            # 종료 조건 확인: 사용자 정의 종료 조건
            done = self.termination_condition(current_state) if self.termination_condition else False

        return total_reward # 시뮬레이션을 통해 얻은 총 보상을 반환

    def backpropagate(self, result):
        # 시뮬레이션 결과를 바탕으로 현재 노드와 상위 노드들의 통계를 업데이트
        # MCTS의 핵심 요소, 각 노드의 방문 횟수와 승리(또는 얻은 보상) 횟수를 업데이트하여 노드의 가치를 재평가
        """
        1. 현재 노드에서 시작하여, 루트 노드까지 이동하며 각 노드의 방문 횟수('visits')와 보상(승리) 횟수('wins')를 업데이트
        2. 시뮬레이션 결과('result'): 시뮬레이션 성공 여부나 얻은 총 보상 여부를 나타냄. 이를 각 노드에 반영
        3. 보상을 업데이트할 때, 해당 시뮬레이션/게임의 특성에 따라 보상의 해석이 달라질 수 있음.(예: 승리/패배, 얻은 점수 등)

        :param result: 시뮬레이션 결과(보상 값). 성공적인 결과일 경우 양수, 그렇지 않은 경우는 음수나 0으로 가정
        """

        # 현재 노드에서 시작하여 루트 노드까지 거슬러 올라감.
        node = self
        while node is not None:
            node.visits += 1 # 현재 노드의 방문 횟수를 1 증가시킴.
            node.wins += result # 시뮬레이션 결과(보상)를 현재 노드의 보상 합계에 추가
            node = node.parent # 부모 노드로 이동

class Agent: 
    """Q-Network와 환경 모델을 사용하여 각 상태에서 최적의 행동을 결정하는 방법 학습하는 에이전트"""
    def __init__(self, state_size, action_size,
                 gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, new_epsilon=None, learning_rate=0.001, new_learning_rate=None,
                 user_defined_items=None, reward_function=None, termination_condition=None,
                 SOME_THRESHOLD=None, SOME_REWARD_THRESHOLD=None, SOME_STATE_THRESHOLD=None,
                 current_episode=None, average_reward=None,
                 state_analysis_callback=None):
        self.state_size = state_size # 상태 vector 크기
        self.action_size = action_size # 가능한 행동 수
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
        self.new_epsilon = new_epsilon
        
        self.learning_rate = learning_rate # 학습률: 에이전트의 학습 속도를 결정 (0 ~ 1)
                                           # 각 학습 단계에서 에이전트의 예측 오류를 얼마나 크게 조정할지 결정함.
                                           # 값이 클수록 학습 중 예측 오류에 대한 조정이 크게 이루어짐.
                                           # 학습률이 너무 높으면 에이전트가 불안정해질 수 있고
                                           # 학습률이 너무 낮으면 학습이 느려지거나 지역 최저점에 갖힐 수 있음.
        self.new_learning_rate = new_learning_rate # 학습률의 동적 조정을 위한 새 학습률

        self.model = QNetwork(state_size, action_size)
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

        self.user_defined_items = user_defined_items
        self.reward_function = reward_function
        self.termination_condition = termination_condition

        self.SOME_THRESHOLD = SOME_THRESHOLD # 에피소드 임계값 (특정 에피소드 이후)
        self.SOME_REWARD_THRESHOLD = SOME_REWARD_THRESHOLD # 특정 보상 임계값 (평균 보상이 일정 수준 이상 도달했을 때)
        self.SOME_STATE_THRESHOLD = SOME_STATE_THRESHOLD # 특정 상태의 임계값

        self.current_episode = current_episode
        self.average_reward = average_reward
        self.state_analysis_callback = state_analysis_callback

        self.env = Environment(state_size, action_size) # 에이전트가 상호작용할 강화학습의 환경을 설정
        
        #환경 모델: 상태와 행동을 입력으로 받아 다음 상태와 예상 보상을 예측함.
        self.env_model = EnvironmentModel(state_size, action_size)
        self.env_model.compile(loss=['mse', 'mse'], optimizer=Adam(learning_rate=learning_rate), metrics=['mae'])

    def plan(self, root_state, iterations=100):
        """
        몬테카를로 트리 서치를 기반으로 최적의 행동 시퀀스를 결정
        :param root_state: 현재 상태
        :param iterations: 탐색을 위한 반복 횟수
        :return: 최적의 행동
        """

        # 루트 노드의 초기화
        root_node = MCTSNode(state=root_state, user_defined_items=self.user_defined_items,
                             reward_function=self.reward_function,
                             termination_condition=self.termination_condition)

        for _ in range(iterations):
            node = root_node
            # 1. Selection: 최적의 자식 노드를 선택하는 과정
            while not node.is_terminal() and node.untried_actions == [] and node.children != []:
                node = node.select_child()
            # 2. Expansion: 탐색하지 않은 새로운 행동으로 노드를 확장하는 과정
            if not node.is_terminal() and node.untried_actions != []:
                action = random.choice(node.untried_actions)
                node = node.expand(action)
            # 3. Simulation: 확장된 노드로부터 시뮬레이션을 실행하여 결과를 예측하는 과정
            result = node.simulate()
            # 4. Backpropagation 역전파: 시뮬레이션 결과를 바탕으로 노드의 정보를 업데이트하는 과정
            node.backpropagate(result)

        # 트리 탐색을 통해 얻은 정볼를 바탕으로 최적의 행동 반환
        best_action = max(root_node.children, key=lambda x: x.wins / x.visits if x.visits > 0 else 0).action
        return best_action

    def remember(self, state, action, reward, next_state, done):
        """에이전트의 경험을 메모리에 저장, 경험은 (상태, (취해진) 행동, (받은) 보상, 다음 상태 (결과), (에피소드) 종료 여부)의 튜플로 저장됨"""
        self.memory.append((state, action, reward, next_state, done))
        # 환경 모델 업데이트를 위한 데이터 준비

    # 미니배치를 사용해 주기적으로 업데이트하는 로직
    def update_environment_model(self, batch_size):
        """환경 모델을 업데이트. 메모리에 저장된 경험을 사용해 환경 모델을 학습함."""
        if len(self.memory) < batch_size:
            return #메모리에 충분한 데이터가 없으면 학습하지 않음.
        
        minibatch = random.sample(self.memory, batch_size) # 미니배치를 선택
        for state, action, reward, next_state, done in minibatch: # 미니배치로부터 데이터 추출

            # 환경 모델은 상태와 행동을 입력으로 받아, 다음 상태와 예상 보상을 출력함.
            # 여기에서는 다음 상태와 보상을 각각의 목표로 하여 환경 모델을 학습시킴.
            self.env_model.fit([np.array([state]), np.array([action])], [np.array([next_state]), np.array([reward])], epochs=1, verbose=0)

    def should_use_mcts(self, state):
        """MCTS를 사용할지 여부를 결정하는 조건식"""
        # 예: 특정 에피소드 이후 또는 평균 보상이 일정 수준에 도달했을 때 MCTS 사용 (true 반환)
        # 이 함수는 사용자가 특정 조건에 따라 구현할 수 있음
        if self.current_episode > self.SOME_THRESHOLD or self.average_reward > self.SOME_REWARD_THRESHOLD:
            return True
        # 상태 정보(state)를 분석하여 결정하는 것도 가능
        if self.state_analysis_callback is not None and self.state_analysis_callback(state):
            return True
        return False
    
    """
    사용 예시:

    사용자는 상태 정보를 기반으로 MCTS 사용 여부를 결정하는 함수를 정의하고, 이를 에이전트의 생성자에 전달할 수 있음.

    def custom_state_analysis(state):
        # 상태 정보를 기반으로 복잡한 분석 수행
        # 예시: 상태 벡터의 특정 값(여기서는 평균값)이 임계값을 초과하는 경우 True 반환
        return state.mean() > SOME_STATE_THRESHOLD

    agent = Agent(state_size=10, action_size=4, ..., state_analysis_callback=custom_state_analysis)
    """

    def act(self, state, training=True):
        """
            에이전트의 행동 결정 로직
            ε -greedy 전략을 사용하여 에이전트의 행동 결정, 무작위로 탐험하거나 학습된 정책을 활용
            use_best_action 플래그가 True일 경우, MCTS를 통해 구한 best_action을 사용

            - training: 학습 중인지 여부를 나타냄. false인 경우, 항상 학습된 정책을 사용
         """
        if not training:
            # 학습 모드가 아닌 경우 항상 학습된 정책 사용
            qs = self.model.predict(np.array([state])) # 현재 상태에 대한 행동 가치 함수 예측
            return np.argmax(qs[0]) # 가장 높은 Q값을 가진 행동 선택

        use_best_action = self.should_use_mcts(state)

        #학습 모드인 경우
        if use_best_action:
            # MCTS를 사용해서 best_action을 계산
            best_action = self.plan(state)
            return best_action
        elif np.random.rand() <= self.epsilon:
             # 무작위 탐험
            return np.random.randint(self.action_size)
        else:
            # 학습된 정책에 따라 행동 선택
            qs = self.model.predict(np.array([state])) # 현재 상태에 대한 행동 가치 함수 예측
            return np.argmax(qs[0]) # 가장 높은 Q값을 가진 행동 선택

    def adjust_learning_rate(self):
        """학습률을 동적으로 조정"""
        K.set_value(self.model.optimizer.learning_rate, self.new_learning_rate)
        K.set_value(self.env_model.optimizer.learning_rate, self.new_learning_rate)

    def adjust_epsilon(self):
        """탐험율을 동적으로 조정"""
        self.epsilon = max(self.epsilon_min, min(1.0, self.new_epsilon))

    def replay(self, batch_size):
        """경험 리플레이를 통한 모델 학습 (저장된 경험을 사용해서 네트워크를 학습하는 역할), 미니배치를 무작위로 선택하여 학습"""
        # 메모리에 충분한 경험이 쌓이지 않았다면 함수를 종료
        if len(self.memory) < batch_size:
            return
        # 경험 메모리에서 무작위로 미니배치를 선택
        minibatch = np.random.choice(self.memory, batch_size, replace=False)
        for state, action, reward, next_state, done in minibatch:
            # 실제 경험을 사용한 학습
            target_real = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            # 환경 모델로부터 예측된 다음 상태와 보상을 사용하여 가상의 타겟을 계산
            predicted_next_state, predicted_reward = self.env_model.predict([state, action])
            target_predicted = predicted_reward if done else predicted_reward + self.gamma * np.amax(self.model.predict(predicted_next_state)[0])

            # 실제 경험과 환경 모델 예측을 통해 얻은 정보를 결합하여 최종 타겟을 결정
            target = (target_real + target_predicted) / 2 # 두 타겟의 평균을 사용 (추후 개선)

            # 현재 상태에 대한 모델의 예측 값을 가져옴
            target_f = self.model.predict(state)
            #선택된 행동에 대한 타겟 값을 업데이트
            target_f[0][action] = target
            # 모델 업데이트
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # 탐험율 감소
        self.update_epsilon()

        # 학습 이후 특정 조건에 따라 학습률, 탐험율 조정 로직
        if self.current_episode > self.SOME_THRESHOLD:
                self.adjust_learning_rate()
                self.adjust_epsilon()

        # 환경 모델 업데이트를 위한 미니배치 학습 (update_environment_model의 주기적인 호출)
        self.update_environment_model(batch_size)

    def evaluate_agent(self):
        """환경 모델을 사용한 시뮬레이션과 실제 환경과의 성능을 비교함."""
        simulated_rewards = []
        real_rewards = []

        # 환경 모델과 실제 환경에서 각각 20회의 에피소드를 실행하고 평균 보상을 비교함.
        for _ in range(20):
            simulated_reward = self.simulate_environment()
            real_reward = self.interact_with_environment()
            simulated_rewards.append(simulated_reward)
            real_rewards.append(real_reward)

        average_simulated_reward = sum(simulated_rewards) / len(simulated_rewards)
        average_real_reward = sum(real_rewards) / len(real_rewards)

        print(f"Average Simulated Reward:: {average_simulated_reward}, Average Real Reward: {average_real_reward}")

    def simulate_environment(self):
        """환경 모델을 사용해 에이전트의 성능을 시뮬레이션"""
        # 환경 모델을 사용해 에이전트의 행동을 시뮬레이션하고 평균 보상을 계산: 기본 형태

        state = self.env.reset() # 환경을 초기 상태로 리셋
        total_reward = 0
        done = False
        while not done:
            action = self.act(state, training=False) # 에이전트의 행동 결정
            next_state, reward, done = self.env_model.predict(state, action) # 환경 모델을 사용해 다음 상태와 보상 예측
            total_reward += reward
            state = next_state
        return total_reward

    def interact_with_environment(self):
        """실제 환경과 상호작용하여 에이전트의 성능을 평가"""
        # 실제 환경에서 에이전트의 행동을 실행하고 평균 보상을 계산: 기본 형태
        state = self.env.reset() # 환경을 초기 상태로 리셋
        total_reward = 0
        done = False
        while not done:
            action = self.act(state, training=False) # 에이전트의 행동 결정
            next_state, reward, done = self.env.step(action) # 실제 환경에서 행동을 실행하고 결과 얻기
            total_reward += reward
            state = next_state
        return total_reward

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