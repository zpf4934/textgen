# -*- coding:utf-8 -*-
"""
# File       : temp.py
# Time       ：2023/10/25 11:09
# Author     ：andy
# version    ：python 3.9
"""
from tqdm import tqdm
import pandas as pd
import numpy as np

red_result = []
blue_result = []
size = 1000
union_lotto = pd.read_csv('union_lotto.csv')
for j in tqdm(range(0, len(union_lotto) - size - 1)):
    union = union_lotto[j: j + size + 1]
    red_ball = union[['red1', 'red2', 'red3', 'red4', 'red5', 'red6']]
    blue_ball = union[['blue']]
    ball_num = pd.DataFrame(pd.DataFrame(red_ball[:-1].values.reshape(-1, 1)).value_counts(normalize=True)).sort_index().T.reset_index(drop=True)
    ball_num.columns = ['proportion' + str(i) for i in range(1, 34)]
    ball_num[['sum']] = red_ball.iloc[-2].sum()
    ball_num[['span']] = red_ball.iloc[-2].max() - red_ball.iloc[-2].min()
    ball_num[['size1', 'size2', 'size3', 'size4', 'size5', 'size6']] = (red_ball.iloc[-2] > 17).values.tolist()
    ball_num[['parity1', 'parity2', 'parity3', 'parity4', 'parity5', 'parity6']] = (red_ball.iloc[-2] % 2 == 0).values.tolist()
    num = []
    for i in range(1, 34):
        ball_num[['num' + str(i)]] = (red_ball[:-1] == i).sum().sum()
    now_miss = []
    avg_miss = []
    continues = []
    for i in range(1, 34):
        miss = []
        miss_num = 0
        continuous = 0
        max_continuous = 0
        for index, row in red_ball[:-1].iterrows():
            if i in row.values:
                continuous += 1
                if miss_num != 0:
                    miss.append(miss_num)
                    miss_num = 0
            else:
                miss_num += 1
                if continuous > max_continuous:
                    max_continuous = continuous
                continuous = 0
        ball_num[['now_miss' + str(i)]] = miss[-1]
        ball_num[['avg_miss' + str(i)]] = sum(miss) // len(miss)
        ball_num[['max_continuous' + str(i)]] = max_continuous
    ball_num = pd.DataFrame(np.repeat(ball_num.values, 6, axis=0), columns=ball_num.columns)
    ball_num['label'] = red_ball.iloc[-1].values
    red_result.append(ball_num.copy())
    blue = blue_ball[:-1].reset_index(drop=True).T
    blue['label'] = blue_ball.iloc[-1].values[0]
    blue_result.append(blue.copy())
red_result = pd.concat(red_result).reset_index(drop=True)
red_result = red_result.mask(red_result == True, 1)
red_result = red_result.mask(red_result == False, 0)
red_result.to_csv('red.csv', index=False)
blue_result = pd.concat(blue_result).reset_index(drop=True)
blue_result.to_csv('blue.csv', index=False)
