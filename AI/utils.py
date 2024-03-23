# utils.py: 학습 과정에서 사용되는 보조 함수(학습 과정 그래프 등)를 포함하는 파일

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_learning_curves(data, titles, filename):
    """
    학습 과정에서 변화하는 여러 지표들을 그래프로 시각화

    :param data: 그래프로 나타낼 데이터의 리스트. 각 요소는 그래프로 나타낼 데이터의 시퀀스.
    :param titles: 각 서브플롯의 제목 리스트
    :param filename: 그래프 이미지를 저장할 파일 경로
    """
    num_plots = len(data)
    plt.figure(figsize=(10, 6 * num_plots)) # 그래프 크기 설정
    for i, (plot_data, title) in enumerate(zip(data, titles), 1):
        plt.subplot(num_plots, 1, i)
        plt.plot(plot_data)
        plt.titlel(title)
        plt.xlabel('Episode')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close() # 현재 그래프 닫기

def smooth_scores(scores, window=10):
    """
    (점수 변화가 너무 급격하거나 노이즈가 심할 경우), 점수(보상)의 이동 평균을 계산하여 부드러운 학습 곡선 생성

    :param scores: 에피소드별 점수 리스트
    :param window: 이동 평균을 계산할 윈도우 크기
    """
    smoothed_scores = pd.Series(scores).rolling(window, min_periods=1).mean().tolist()
    return smoothed_scores

def plot_and_save_all_curves(rewards, epsilons, learning_rates, steps, filename):
    """
    보상, ε값의 변화, 학습률의 변화, 스텝 수 등의 학습 과정 지표를 시각화하고 저장

    :param rewards: 에피소드별 총 보상 리스트
    :param epsilons: 에피소드별 ε값 리스트
    :param learning_rates: 에피소드별 학습률 리스트
    :param steps: 에피소드별 스텝 수 리스트
    :param filename: 그래프 이미지를 저장할 파일 이름
    """
    rewards_smoothed = smooth_scores(rewards)
    data = [rewards_smoothed, epsilons, learning_rates, steps]
    titles = ['Rewards (Smoothed)', 'Epsilon Values', 'Learning Rates', 'Steps per Episode']
    plot_learning_curves(data, titles, filename)