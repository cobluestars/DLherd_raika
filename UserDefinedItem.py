# UserDefinedItem.py

import numpy as np
from datetime import datetime, timedelta
from scipy.stats import beta, lognorm, expon, binom
import random

class UserDefinedItem:
    """
    커스텀 데이터 항목을 나타내는 클래스
    다양한 확률 분포를 사용해 값 생성이 가능
    """
    def __init__(self, name, item_type, distribution='uniform', mean=None, std_dev=None, mode=None, median=None, weighted_mean=None, geometric_mean=None,
                 randomizeArrays=False, arraySelectionCount=1,
                 randomizeObjects=False, objectSelectionCount=1,  
                 randomizeSelectionCount=False, selectionProbability=False,
                 options=None, probability_settings=None,
                 peak_times=None,
                 contextBasedOptions=None):
        """
        UserDefinedItem 인스턴스를 초기화

        :param name: 항목명 (str)
        :param item_type: 데이터 타입 ('time', 'number', 'string', 'boolean', 'array', 'object')
        :param options: 분포 생성에 사용될 옵션 (tuple, list 등)
        
        :param peak_times: 피크 타임 설정 (['피크 타임 시작', '피크 타임 종료', '전체 시간 100% 기준으로 피크 타임 내에서 타임스탬프가 찍힐 확률 (미기재 시 0%)']: ['2024-04-01T06:00:00', '2024-04-01T08:00:00', 0.25])

        :param probability_settings: 배열의 요소 선택에 사용될 확률 설정
            # 예제
            options = ["apple", "banana", "cherry"]
            probability_settings = [
                {"identifier": "apple", "probability": 20},
                {"identifier": "banana", "probability": 30},
                {"identifier": "cherry", "probability": 50}
            ]       

        :param randomizeArrays: 배열 항목의 랜덤 선택 활성화 여부
        :param arraySelectionCount: 배열에서 선택할 항목 수
        :param randomizeObjects: 객체 항목의 랜덤 선택 활성화 여부
        :param objectSelectionCount: 객체에서 선택할 항목 수

        :param randomizeSelectionCount: 선택한 항목 수 내에서 무작위 선택 활성화 여부 (ex: 3개 선택 시, 1~3개 랜덤 선택)
        :param selectionProbaility: 선택 확률 조정 활성화 여부

        :param contextBasedOptions: 특정 컨텍스트에 기반한 조건부 옵션

        :param distributionn: 확률 분포 타입('uniform': 완전 랜덤, 'normal': 정규 분포, 'beta': 베타 분포, 'log-normal', 'exponential', 'binomial')
        :param mean: 정규 분포의 평균 (float, normal에만 적용)
        :param std_dev: 정규 분포의 표준편차 (float, normal에만 적용)

        🐺데이터 전처리 및 머신러닝(강화학습)에 활용될 값들🐺
        :param mode: 최빈값 (float, optional)
        :param median: 중앙값 (float, optional)
        :param weighted_mean: 가중평균 (float, optional)
        :param geometric_mean: 기하평균 (float, optional)

        """
        self.name = name
        self.type = item_type
        self.peak_times = peak_times if peak_times else []
        self.distribution = distribution
        self.mean = mean
        self.std_dev = std_dev
        self.mode = mode
        self.median = median
        self.weighted_mean = weighted_mean
        self.geometric_mean = geometric_mean
        self.options = options
        self.probability_settings = probability_settings if probability_settings else []
        self.randomizeArrays = randomizeArrays
        self.arraySelectionCount = arraySelectionCount
        self.randomizeObjects = randomizeObjects
        self.objectSelectionCount = objectSelectionCount
        self.randomizeSelectionCount = randomizeSelectionCount
        self.selectionProbability = selectionProbability
        self.contextBasedOptions = contextBasedOptions

    def evaluate_context_based_options(self, context):
        """
        특정 컨텍스트에 기반하여 조건부 옵션 값을 계산
        """
        if self.contextBasedOptions:
            return self.contextBasedOptions(context)
        return None

    def update_parameters(self, **kwargs):  # *^kwargs: 키워드 인자, update_parameters 메소드는 키워드 인자를 무제한으로 받아들임.
        """
        UserDefinedItem 인스턴스의 매개변수를 동적으로 업데이트하는 메소드
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{self.__class__.__name__}에 '{key}'라는 속성이 없습니다.: {self.__class__.__name__} does not have an attribute named '{key}'")

    def setting_probabilities(self):
        """
        설정된 확률(probability_settings)에 따라, 옵션에서의 각 요소들이 선택될 확률을 계산

        :return: 옵션에서의 각 요소들이 선택될 확률 리스트
        """
        probabilities = [0] * len(self.options)
        total_probability_assigned = 0

        # 설정된 확률을 기반으로 probabilities 배열을 채움
        for setting in self.probability_settings:
        # setting: self.probability_settings 리스트의 각 항목을 순회하는 데 사용되는 임시 변수, identifier와 probability 키를 가진 딕셔너리
            if setting['identifier'] in self.options:
                index = self.options.index(setting['identifier'])
                probabilities[index] = setting['probability']
                total_probability_assigned += setting['probability']

        # 확률이 설정되지 않은 항목들에 대해 나머지 확률을 균등 분배
        if total_probability_assigned < 100:
            remaining_probability = 100 - total_probability_assigned
            unassigned = [i for i, p in enumerate(probabilities) if p == 0]
            #todo: i, p의 역할과 enumerate함수의 역할 알아보기
            for i in unassigned:
                probabilities[i] = remaining_probability / len(unassigned)

        return probabilities

    def apply_probability_based_selection(self, probabilities):
        """
        확률 기반으로 옵션 요소를 선택

        :param probabilities: 옵션에서의 각 요소들이 선택될 확률 리스트
        :return: 선택된 옵션 (요소)
        """
        if not self.probability_settings:
            #확률 설정이 없으면 무작위 선택
            return random.choice(self.options)

        selected_indexes = []
        for index, (option, probability) in enumerate(zip(self.options, probabilities)):
            if random.random() * 100 < probability:
                selected_indexes.append(index)

        # 선택된 인덱스가 없다면 무작위로 하나를 선택
        if not selected_indexes:
            selected_indexes.append(random.randint(0, len(self.options) - 1))

        return selected_indexes # 선택된 모든 옵션의 인덱스를 반환

    def generate_value(self, context=None):
        """
        설정된 확률 분포에 따라 값 생성

        :param context: 컨텍스트 데이터, 조건부 로직 실행에 사용됨.
        :return: 생성된 값 (float, int, str 등 type에 따라 다름)
        """

        # 조건부 옵션 처리
        if self.contextBasedOptions:
            modified_options = self.contextBasedOptions(context)
            if modified_options:
                if 'options' in modified_options:
                    self.options = modified_options['options']
                if 'name' in modified_options:
                    self.name = modified_options['name']
                if 'type' in modified_options:
                    self.type = modified_options['type']
                if 'peak_times' in modified_options:
                    self.peak_times = modified_options['peak_times']
                if 'distribution' in modified_options:
                    self.distribution = modified_options['distribution']
                if 'mean' in modified_options:
                    self.mean = modified_options['mean']
                if 'std_dev' in modified_options:
                    self.std_dev = modified_options['std_dev']
                if 'mode' in modified_options:
                    self.mode = modified_options['mode']
                if 'median' in modified_options:
                    self.median = modified_options['median']
                if 'weighted_mean' in modified_options:
                    self.weighted_mean = modified_options['weighted_mean']
                if 'geometric_mean' in modified_options:
                    self.geometric_mean = modified_options['geometric_mean']
                if 'probability_setting' in modified_options:
                    self.probability_settings = modified_options['probability_setting']
                if 'randomizeArrays' in modified_options:
                    self.randomizeArrays = modified_options['randomizeArrays']
                if 'arraySelectionCount' in modified_options:
                    self.arraySelectionCount = modified_options['arraySelectionCount']
                if 'randomizeObjects' in modified_options:
                    self.randomizeObjects = modified_options['randomizeObjects']
                if 'objectSelectionCount' in modified_options:
                    self.objectSelectionCount = modified_options['objectSelectionCount']
                if 'randomizeSelectionCount' in modified_options:
                    self.randomizeSelectionCount = modified_options['randomizeSelectionCount']
                if 'selectionProbability' in modified_options:
                    self.selectionProbability = modified_options['selectionProbability']
                if 'contextBasedOptions' in modified_options:
                    self.contextBasedOptions = modified_options['contextBasedOptions']

        if self.type == 'time':
        #시간 타입
            # options가 now or 특정 시간 문자열일 경우, 해당 시간 반환
            if isinstance(self.options, str):
                if self.options == 'now':
                    return datetime.now()
                else:
                    return datetime.fromisoformat(self.options)
                
            # options가 리스트 [시작 시간, 종료 시간] 형태일 경우,
            elif isinstance(self.options, list):
                start_time = datetime.fromisoformat(self.options[0])
                end_time = datetime.fromisoformat(self.options[1])

                if start_time > end_time:
                    raise ValueError("시작 시간이 종료 시간보다 미래일 수 없습니다.")
                    return datetime.now().isoformat()
                
                total_seconds = int((end_time - start_time).total_seconds())

                # 피크 타임 확률 계산
                specified_peak_probabilities = [peak[2] for peak in self.peak_times if len(peak) > 2]
                total_peak_probability = sum(specified_peak_probabilities)

                if total_peak_probability > 1:
                    raise ValueError("피크 타임 확률의 합이 100%를 초과합니다.")

                # 비피크 타임 확률 계산
                non_peak_probability = 1 - total_peak_probability

                # 전체 시간 범위 내 랜덤 타임스탬프 생성 로직
                random_value = np.random.rand()
                accumulated_probability = 0

                # 피크 타임 처리
                for peak in self.peak_times:
                    peak_start, peak_end = datetime.fromisoformat(peak[0]), datetime.fromisoformat(peak[1])
                    peak_duration = (peak_end - peak_start).total_seconds()
                    peak_probability = peak[2]
                    weighted_peak_probability = (peak_duration / total_seconds) * peak_probability * 8.5 # todo: 8.5 ~ 10배 곱해줘야 값이 제대로 나오는 이유 알아내기.
                    accumulated_probability += weighted_peak_probability
                    if random_value < accumulated_probability:
                        # 피크 타임 내 랜덤 시간 선택
                        random_second_within_peak = random.randint(0, int(peak_duration))
                        return (peak_start + timedelta(seconds=random_second_within_peak)).isoformat()

                # 비피크 타임 처리
                current_time = start_time
                for peak in sorted(self.peak_times, key=lambda x: datetime.fromisoformat(x[0])):
                    peak_start, peak_end = datetime.fromisoformat(peak[0]), datetime.fromisoformat(peak[1])
                    if current_time < peak_start:
                        non_peak_duration = (peak_start - current_time).total_seconds()
                        weighted_non_peak_probability = (non_peak_duration / total_seconds) * non_peak_probability
                        accumulated_probability += weighted_non_peak_probability
                        if random_value < accumulated_probability:
                            # 비피크 타임 내 랜덤 시간 선택
                            random_second = random.randint(0, int(non_peak_duration))
                            return (current_time + timedelta(seconds=random_second)).isoformat()
                    current_time = max(current_time, peak_end)

                if current_time < end_time:
                    non_peak_duration = (end_time - current_time).total_seconds()
                    weighted_non_peak_probability = (non_peak_duration / total_seconds) * non_peak_probability
                    accumulated_probability += weighted_non_peak_probability
                    if random_value < accumulated_probability:
                        # 비피크 타임 내 랜덤 시간 선택
                        random_second = random.randint(0, int(non_peak_duration))
                        return (current_time + timedelta(seconds=random_second)).isoformat()

                # 주어진 확률에 따라 타임스탬프를 생성하지 못한 경우, 기본값 반환
                return datetime.now().isoformat()
    
            else:
                raise ValueError("올바르지 않은 시간 옵션입니다.")
                return datetime.now().isoformat()

        if self.type == 'number':
        #숫자 타입
            # options가 단일 숫자일 경우, 해당 숫자 반환
            if isinstance(self.options, (int, float)):
                return self.options

            # 확률 분포에 따른 값 생성 로직
            if self.distribution == 'uniform':
                #완전 랜덤한 값 생성(uniform distribution)
                return np.random.uniform(self.options[0], self.options[1])
            elif self.distribution == 'normal':
                #정규 분포에서 값 생성(gaussian distribution)
                if self.mean is None or self.std_dev is None:
                    raise ValueError("Normal distribution requires 'mean' and 'std_dev' values.")
                return np.random.normal(self.mean, self.std_dev)
            elif self.distribution == 'beta':
                #베타 분포에서 값 생성(beta distribution): 0과 1 사이 값을 갖는 분포로, 두 매개변수 a와 b에 의해 모양이 결정됨.
                a, b = self.options
                return beta(a, b).rvs()
            elif self.distribution == 'log-normal':
                #Log-normal distribution: 변량의 로그가 정규 분포를 이루는 분포. 주로 비대칭적 데이터에 사용됨.
                return lognorm(s=self.std_dev, scale=np.exp(self.mean)).rvs()
            elif self.distribution == 'exponential':
                #Exponential distribution: 사건 간의 시간을 모델링할 때 사용하는 연속 확률 분포
                return expon(scale=1/self.mean).rvs()
            elif self.distribution == 'binomial':
                #Binomial distribution: 고정된 수의 독립 시행에서 성공 횟수를 나타내는 이산 확률 분포, n: 시행 횟수, p: 각 시행에서 성공할 확률
                n, p = self.options
                return binom(n=n, p=p).rvs()
            elif self.distribution == 'custom':
                # 특정 상황에 따라 다른 통계적 지표를 사용하여 값 생성
                if context and context.get('use_mode', False) and self.mode is not None:
                    # 컨텍스트에 따라 mmode 사용
                    return self.mode
                elif context and context.get('use_median', False) and self.median is not None:
                    # 컨텍스트에 따라 median 사용
                    return self.median
                elif context and context.get('use_weighted_mean', False) and self.weighted_mean is not None:
                    # 컨텍스트에 따라 weighted_mean 사용
                    return self.weighted_mean
                elif context and context.get('use_geometric_mean', False) and self.geometric_mean is not None:
                    # 컨텍스트에 따라 geometric_mean 사용
                    return self.geometric_mean
            else:
                raise ValueError(f"지원하지 않는 확률분포 타입입니다. Unsupported distribution type: {self.distribution}")

        elif self.type == 'string':
        #문자열 타입
            # 단일 문자열 처리
            if isinstance(self.options, str):
                return self.options
            # 문자열 배열 처리
            elif isinstance(self.options, list) and all(isinstance(option, str) for option in self.options):
                # 문자열 배열인 경우, 완전 랜덤 or 확률 기반으로 하나를 선택
                probabilities = self.setting_probabilities()
                return self.apply_probability_based_selection(probabilities)
            else:
                raise ValueError(f"options에 문자열 혹은 문자열 배열을 입력해주세요. Please provide a string or a list of strings in options.")

        elif self.type == 'boolean':
        #boolean 타입
            #boolean: 50% 확률로 true or false 반환
            return random.choice([True, False])

        elif self.type == 'array':
        #배열 타입
            if isinstance(self.options, list):
                # options 리스트의 모든 요소가 유효 타입(재귀 처리를 위한 UserDefinedItem 인스턴스 포함)인지 확인
                if all(isinstance(option, (UserDefinedItem, list, str, int, float, bool)) for option in self.options):
                    result = [] # 결과 리스트를 초기화

                    selected_indexes = range(len(self.options))  # 모든 인덱스를 선택

                    # 배열 요소를 선택해야 하는 경우
                    if self.randomizeArrays:
                        # 확률 기반 선택(selectionProbability)가 활성화되어 있고 확률 설정(probability_setting)이 제공된 경우
                        if self.selectionProbability and self.probability_settings:
                            probabilities = self.setting_probabilities()
                            selected_indexes = self.apply_probability_based_selection(probabilities)
                        else:
                            #특정 확률 없이 완전 무작위 선택을 위한 경우
                            selected_indexes = range(len(self.options))

                    # 설정에 따라 선택할 항목의 수(selected_count)를 결정
                    selected_count = min(len(selected_indexes), self.arraySelectionCount)
                    if self.randomizeSelectionCount:
                        selected_count = random.randint(1, selected_count) # 1 ~ selected_count 사이에서 랜덤한 수(정수) 1개 선택
                    selected_indexes = random.sample(list(selected_indexes), selected_count) # selected_indexes 리스트 중 selected_count개 선택

                else:
                    # 배열을 무작위로 선택하지 않는 경우, 모든 옵션을 사용
                    selected_indexes = range(len(self.options))

                if self.randomizeArrays == True:
                    # 배열에서 선택된 인덱스에 해당하는 항목을 처리
                    for i in selected_indexes:
                        option = self.options[i]
                        # 옵션이 UserDefinedItem 인스턴스인 경우, 그 인스턴스의 generate_value 메서드를 재귀적으로 호출
                        if isinstance(option, UserDefinedItem):
                            generated_value = option.generate_value()
                            # 여기에서 결과를 구조화히여 추가
                            result.append({option.name: generated_value})
                        else:
                            # 리터럴(정수, 문자열, 부동소수점 등)은 직접 결과 리스트에 추가
                            result.append(option)
                    return result
                else:
                    for option in self.options: # 배열에서 모든 인덱스를 처리
                        if isinstance(option, UserDefinedItem):
                            generated_value = option.generate_value()
                            # option.name을 키로 사용하고, generated_value를 값으로 사용하여 딕셔너리 생성
                            result.append({option.name: generated_value})
                            # 'array' 타입의 이름을 키로, 결과 리스트를 값으로 사용하여 딕셔너리 반환
                        else:
                            result.append(option)
 
                return result
            else:
                raise ValueError("'array' 타입의 'options'는 UserDefinedItem 인스턴스나 리터럴 값이 포함된 리스트여야 합니다.: Options for 'array' type must be a list of UserDefinedItem instances or literals.")

        elif self.type == 'object':
        #객체 타입
            if isinstance(self.options, dict):
                # 객체 내 속성을 무작위로 선택하는 경우
                if self.randomizeObjects:
                    # 확률 기반 선택(selectionProbability)이 활성화되어 있고, 확률 설정(probability_settings)이 제공된 경우
                    selected_keys = list(self.options.keys())
                    if self.selectionProbability and self.probability_settings:
                        probabilities = self.setting_probabilities()
                        selected_keys = self.apply_probability_based_selection(probabilities)

                    # 설정에 따라 선택된 속성의 수(selected_count) 결정
                    selected_count = min(len(selected_keys), self.objectSelectionCount)
                    if self.randomizeSelectionCount:
                        selected_count = random.randint(1, selected_count) # 1 ~ selected_count 사이에서 랜덤한 수(정수) 1개 선택
                    selected_keys = random.sample(selected_keys, selected_count)  # selected_keys 중 selected_count개 선택
                else:
                    # 모든 속성을 포함하는 경우
                    selected_keys = list(self.options.keys())

                result = {}

                if self.randomizeObjects == True:
                    for key in selected_keys:
                        option = self.options[key]
                        # 옵션이 UserDefinedItem 인스턴스인 경우, 그 인스턴스의 generate_value 메서드를 재귀적으로 호출
                        if isinstance(option, UserDefinedItem):
                            generated_value = option.generate_value()
                            result[key] = option.generate_value()
                        else:
                            #리터럴(정수, 문자열, 부동소수점 등)은 직접 결과 객체에 추가
                            result[key] = option
                else:
                    for key, option in self.options.items(): # 모든 객체의 속성을 포함
                        if isinstance(option, UserDefinedItem):
                            # UserDefinedItem 인스턴스에 대해서는 generate_value 메서드를 호출
                            generated_value = option.generate_value()
                            result[key] = generated_value
                        else:
                            # 리터럴 값은 직접 결과 객체에 추가
                            result[key] = option
                return result  
            else:
                raise ValueError("'object' 타입의 'options'는 딕셔너리로 제공되어야 합니다.: Options for 'object' type must be a dict.")