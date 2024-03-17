# model.py: 신경망 모델을 정의하는 파일

# Q-Learning: 강화학습(RL)의 한 형태
# 에이전트가 환경과 상호작용하며 학습하는 과정에서 최적의 행동 정책을 찾아가는 방법
# (행동)정책: 주어진 상태에서 어떤 행동을 선택할지를 결정하는 규칙
# 에이전트는 현재 상태에서 어떤 행동을 취했을 때 얻을 수 있는 예상 보상의 크기를 나타내는 Q값(Q-Value)을 기반으로 행동을 결정
# 이러한 Q값을 업데이트하며 최적의 행동 정책을 학습함.

# Q-Learning의 핵심 아이디어: "Q-함수"를 사용하여 주어진 상태에서 각 행동의 예상되는 "품질"을 평가하는 것
# 품질: 특정 행동을 취했을 때 받을 수 있는 미래 보상의 합계에 대한 예측

"""
Q-Learning의 기본적인 업데이트 규칙은 다음과 같은 벨만 방정식에 기초함:

Q(s,a) ← Q(s,a) + α[r + γ MAXa' Q(s',a') - Q(s,a)]

Q(s,a): 상태 s에서 행동 a를 취했을 때 얻을 수 있는 총 보상의 추정치
α: 학습률, 새로운 정보를 얼마나 빠르게 학습할지를 결정
r: 취한 행동으로 인해 즉시 얻는 보상
Υ: 할인율, 미래의 보상을 현재 가치로 환산할 때 사용하는 계수
MAXa' Q(s',a'): 다음 상태에서 가능한 모든 행동에 대한 Q-값 중 최대값

에이전트가 강화를 통해 핛습하면서, Q(s,a)값을 점차 실제 값에 근접하게 조정해 나가는 과정
"""

# Watkins, C.J.C.H. and Dayan, P., 1992. Q-learning. Machine learning, 8(3-4), pp.279-292.

import tensorflow as tf

class QNetwork(tf.keras.Model):
    """신경망을 사용해 Q-Value를 추정하는 모델 정의, 다양한 타입의 데이터 처리 가능"""
    def __init__(self, input_shape, action_size, hidden_units=[24, 24]):
        """
        input_shape: 입력 데이터의 형태. 예컨데, (state_sizes,) 혹은 UserDefinedItem에서 제공하는 데이터 형태

        action_size: 가능한 행동의 수 
                     에이전트가 취할 수 있는 다른 액션의 개수

        hidden_units: 각 은닉층의 뉴런 수를 나타내는 리스트
                      은닉층의 수와 각 은닉층의 뉴런 수를 동적으로 결정하는 것이 가능
        """
        super(QNetwork, self).__init__()
        """
        QNetwork 클래스가 tf.keras.Model 클래스를 상속 받았기에,
        부모 클래스의 생성자를 호출하여 초기화함. 이와 같은 방법으로
        tf.keras.Model의 모든 속성과 메서드를 QNetwork에서도 사용할 수 있게 해줌.
        """
        self.hidden_layers = []
        """
        hidden_units 리스트에 지정된 대로 은닉층과 각 은닉층의 뉴런 수를 동적으로 생성
        hidden_units 리스트를 통해 다양한 구조의 신경망을 설정 가능
        예컨데 [24, 48, 24]와 같이 지정하면 첫 번째 은닉층에 24개의 뉴런, 두 번째 은닉층에 48개의 뉴런, 세 번째 은닉층에 24개의 뉴런을 가진 신경망을 생성 가능
        """
        for idx, units in enumerate(hidden_units):
            if idx == 0:
                #첫 번째 은닉층에서만 input_shape를 정의하여 입력 데이터의 형태 정의
                self.hidden_layers.append(tf.keras.layers.Dense(units, activation='relu', input_shape=input_shape))
            else:
                self.hidden_layers.append(tf.keras.layers.Dense(units, activation='relu'))
        self.output_layer = tf.keras.layers.Dense(action_size, activation='linear')

        """
        기본 형태 예시

        state_size: 상태의 차원
                    강화학습에서 에이전트가 관측할 수 있는 환경의 상태를
                    벡터로 나타낼 때의 크기

        self.dense1 = tf.keras.layers.Dense(24, activation='relu', input_shape=(state_size,))
        self.dense2 = tf.keras.layers.Dense(24, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_size, activation='linear')
        
        self.dense1: 첫 번째 Dense(완전 연결)층을 정의함. 
                     이 층은 24개의 뉴런을 가지고, 활성화 함수로 ReLU를 사용
                     input_shape=(state_size,)는 이 신경망의 입력 차원이 state_size임을 나타냄.
                     에이전트가 관측할 수 있는 환경의 상태 벡터의 크기와 일치

        self.dense2: 두 번째 Dense층을 정의함.
                     이 층도 24개의 뉴런을 가지고, 활성화 함수로 ReLU를 사용
                     이전 층의 출력을 입력으로 받음.

        self.output_layer: 출력 층을 정의
                           'action_size'만큼의 뉴런을 가지며,
                           각 뉴런은 가능한 각 액션에 대응하는 Q-Value를 출력
                           활성화 함수로는 선형 함수를 사용. 이는 각 액션의 예상되는 보상을 직접적으로 추정하기 위함.
        """

    def call(self, inputs):
        """입력된 상태에 대해 각 액션의 Q-Value를 반환"""
        x = inputs
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        return self.output_layer(x)

    """
    이러한 구조를 통해, Q-Network모델은 주어진 상태에 대해 모든 가능한 행동의 Q-Value를 출력할 수 있음.
    이 정보는 강화학습에서 에이전트가 어떤 행동을 취할지 결정하는 데에 사용됨.
    """