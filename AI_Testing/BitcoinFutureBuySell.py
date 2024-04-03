import json
import random
import sys
import pandas as pd
import ccxt
from datetime import datetime

# Python 모듈 검색 경로에 디렉토리 추가
sys.path.append('c:\\DLherd_raika')

from UserDefinedItem import UserDefinedItem

class MarketDataItem(UserDefinedItem):
    def generate_value(self, context=None):

       # 시장 데이터에 대한 커스텀 처리 로직을 구현
        if self.type == 'market_data':
            market_data = context['market_data']
            # 모든 시간 간격에 대한 데이터 처리
            result = {}
            for timeframe, data in market_data.items():
                result[timeframe] = {
                    'close': data['close'],
                    'high': data['high'],
                    'low': data['low'],
                    'volume': data['volume']
                }
            return result
        else:
            return super().generate_value(context)

# 컨텍스트에 기반한 조건부 옵션 예시 함수
def context_based_price_for_Bitcoin_Sell(context):
    if context is not None and 'Trade Time' in context and context['Trade Time'] == ['2024-04-02T03:00:00', '2024-04-02T04:00:00']:
        return {"name": "Size", "options": [0.001, 0.01], "distribution": "normal", "mean": 0.005, "std_dev": 0.0001}
    else:
        return {"name": "Size", "options": [0.001, 0.005], "distribution": "normal", "mean": 0.0025, "std_dev": 0.0001}

def context_based_price_for_Bitcoin_Buy(context):
    if context is not None and 'Trade Time' in context and context['Trade Time'] == ['2024-04-01T22:00:00', '2024-04-02T23:00:00']:
        return {"name": "Size", "options": [0.001, 0.01], "distribution": "normal", "mean": 0.005, "std_dev": 0.0001}
    else:
        return {"name": "Size", "options": [0.001, 0.005], "distribution": "normal", "mean": 0.0025, "std_dev": 0.0001}

# 커스텀 데이터 항목 인스턴스 생성
user_defined_items = [
    UserDefinedItem(
        name='Bitcoin Buy/Sell',
        item_type='array',
        options=[
            UserDefinedItem(
                name='Bitcoin Sell',
                item_type='array',
                options=[
                    UserDefinedItem(
                        name='Sell Time',
                        item_type='time',
                        options=['now', 1, 'hours'],
                    ),
                    UserDefinedItem(
                        name='Size',
                        item_type='number',
                        contextBasedOptions=context_based_price_for_Bitcoin_Sell
                    )
                ]
            ),
            UserDefinedItem(
                name='Bitcoin Buy',
                item_type='array',
                options=[
                    UserDefinedItem(
                        name='Buy Time',
                        item_type='time',
                        options=['now', 1, 'hours'],
                    ),
                    UserDefinedItem(
                        name='Size',
                        item_type='number',
                        contextBasedOptions=context_based_price_for_Bitcoin_Buy
                    )
                ]
            )
        ],
        randomizeArrays=True,
        selectionProbability=True,
        probability_settings=[
            {"identifier": "Bitcoin Sell", "probability": 70},
            {"identifier": "Bitcoin Buy", "probability": 30}
        ]
    ),
    MarketDataItem(
        name='Market Data',
        item_type='market_data',
        options=['close', 'high', 'low', 'volume'],  # 이 예시에서는 options 사용을 예시로 듭니다.
        distribution='custom'
    )
]

def fetch_market_data():
    timeframes = ['1m', '3m', '5m', '15m', '1h', '4h', '1d', '1w', '1M']
    market_data = {}

    # ccxt 라이브러리를 사용하여 binance 거래소 연결
    exchange = ccxt.binance()
    for timeframe in timeframes:
        # OHLCV 데이터 가져오기
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        market_data[timeframe] = df.iloc[-1].to_dict()  # 최근 데이터 반환

    return market_data

# 데이터 생성 함수 수정
def generate_data(user_defined_items):
    data = []
    
    # 시장 데이터 가져오기
    market_data = fetch_market_data()
    
    for _ in range(100):
        record = {}
        for item in user_defined_items:
            # 각 UserDefinedItem 인스턴스에 대해 컨텍스트(기준 설정)에 기반한 값 생성
            # 각 시간 간격의 시장 데이터를 context에 포함
            context = { 'market_data': market_data }
            value = item.generate_value(context=context)  # 시장 데이터를 컨텍스트로 전달
            record[item.name] = value
        data.append(record)
    return data

# # 데이터 생성
# def generate_data():
#     data = []
    
#     for _ in range(1000):
#         record = {}
#         for item in user_defined_items:
#             # 각 UserDefinedItem 인스턴스에 대해 컨텍스트(기준 설정)에 기반한 값 생성
#             value = item.generate_value()
#             record[item.name] = value    
#         data.append(record)
#     return data

# 생성된 데이터를 json 파일로 저장
def save_data_to_json(data, filename='db.json'):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

# 메인 함수
if __name__=="__main__":
    # 데이터 생성
    data = generate_data(user_defined_items)
    #생성된 데이터를 Json파일로 저장
    save_data_to_json(data)
    print("데이터가 성공적으로 생성되어 {} 파일에 저장되었습니다.".format("db.json"))