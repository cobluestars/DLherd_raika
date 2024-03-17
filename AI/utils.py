# utils.py: 학습 과정에서 사용되는 보조 함수(학습 과정 그래프 등)를 포함하는 파일

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_learning_curve(scores, filename, x=None, title='learning Curve'):
    """
    학습 과정에서, 각 에피소드가 얻은 점수(보상) 변화를 그래프로 시각화
    scores: 에피소드별 점수 리스트
    filename: 그래프 이미지를 저장할 파일 경로
    x: 에피소드 번호 리스트
    title: 그래프 제목
    """
    if x is None:
        x = [i+1 for i in range(len(scores))]
    plt.figure(figsize=(10, 6)) # 그래프 크기 설정
    plt.plot(x, scores, label='Scores')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(filename)
    plt.close() # 현재 그래프 닫기

def smooth_scores(scores, window=10):
    """
    (점수 변화가 너무 급격하거나 노이즈가 심할 경우), 점수(보상)의 이동 평균을 계산하여 부드러운 학습 곡선 생성
    window: 이동 평균을 계산할 윈도우 크기
    """
    smoothed_scores = np.convolve(scores, np.ones(window)/window, 'valid')
    return smoothed_scores

def plot_smoothed_learning_curve(scores, filename, window=10, **kwargs):
    """
    부드러운 학습 곡선을 그래프로 표시
    window: 이동 평균을 계산할 윈도우 크기
    **kwargs: plot_learning_curve 함수에 전달할 추가 인자들
    """
    smoothed_scores = smooth_scores(scores, window)
    plot_learning_curve(smoothed_scores, filename, **kwargs)

def save_model_weights(model, filename):
    """학습된 모델의 가중치를 파일로 저장"""
    model.save_weights(filename)

def load_model_weights(model, filename):
    """파일에서, 학습된 모델의 가중치를 로드"""
    model.load_weights(filename)

# 데이터 분석을 위한 추가 함수
def analyze_data(data):
    """
    주어진 데이터의 기본적인 통계 분석을 수행
    data: 분석할 데이터 (Numpy 배열 또는 Pandas DataFrame)
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    print(data.describe())