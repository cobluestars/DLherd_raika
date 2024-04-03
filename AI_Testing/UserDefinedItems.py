# UserDefinedItems.py

import json
import random
import sys
from datetime import datetime

# Python 모듈 검색 경로에 디렉토리 추가
sys.path.append('c:\\DLherd_raika')

from UserDefinedItem import UserDefinedItem

# 컨텍스트에 기반한 조건부 옵션 예시 함수
def context_based_salary_for_student(context):
    from random import random
    if random() < 0.0001:  # 0.01% 확률
        return {"options": [20000, 100000]}

def context_based_salary_for_developer(context):
    if context is not None and 'age' in context and context['age'] < 30:
        return {"name": "salary", "options": [20000, 40000], "distribution": "normal", "mean": 27000, "std_dev": 6}
    else:
        return {"name": "salary", "options": [30000, 100000], "distribution": "normal", "mean": 40000, "std_dev": 6}

def context_based_salary_for_accountant(context):
    if context is not None and 'age' in context and context['age'] < 30:
        return {"name": "salary", "options": [25000, 40000], "distribution": "normal", "mean": 30000, "std_dev": 6}
    else:
        return {"name": "salary", "options": [30000, 100000], "distribution": "normal", "mean": 40000, "std_dev": 6}

# 커스텀 데이터 항목 인스턴스 생성
user_defined_items = [
    UserDefinedItem(
        name='TimeStamp',
        item_type='time',
        options=[12, 'now', 'hours'],
        peak_times=[
        ['2024-04-03T00:45:00', '2024-04-03T01:00:00', 0.3],
        # ['2024-04-01T03:00:00', '2024-04-01T04:00:00', 0.2],
        ['2024-04-03T07:00:00', '2024-04-03T08:00:00', 0.5]
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
    ),
]

# 데이터 생성
def generate_data():
    data = []
    peak_times = user_defined_items[0].peak_times
    peak_time_counts = {(peak[0], peak[1]): 0 for peak in peak_times}
    
    for _ in range(10):
        record = {}
        for item in user_defined_items:
            # 각 UserDefinedItem 인스턴스에 대해 컨텍스트(기준 설정)에 기반한 값 생성
            value = item.generate_value()
            record[item.name] = value
    
              # 타임스탬프가 피크 타임에 속하는지 확인하고 카운트
            if item.name == 'TimeStamp' and isinstance(value, str):
                try:
                    timestamp = datetime.fromisoformat(value)
                    for peak in peak_times:
                        start, end = datetime.fromisoformat(peak[0]), datetime.fromisoformat(peak[1])
                        if start <= timestamp <= end:
                            peak_time_counts[(peak[0], peak[1])] += 1        
                except ValueError:
                    print(f"Invalid date format: {value}")
    
        data.append(record)
    return data, peak_time_counts

# 생성된 데이터를 json 파일로 저장
def save_data_to_json(data, filename='db.json'):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

# 메인 함수
if __name__=="__main__":
    # 데이터 생성
    data, peak_time_counts = generate_data()
    #생성된 데이터를 Json파일로 저장
    save_data_to_json(data)
    print("데이터가 성공적으로 생성되어 {} 파일에 저장되었습니다.".format("db.json"))
    for peak, count in peak_time_counts.items():
        print(f"피크 타임 {peak[0]} ~ {peak[1]}에는 {count}개의 데이터가 생성됨.")