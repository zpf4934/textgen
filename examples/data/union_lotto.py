# -*- coding:utf-8 -*-
"""
# File       : union_lotto.py
# Time       ：2023/9/6 14:19
# Author     ：andy
# version    ：python 3.9
"""
import json

import pandas as pd

lotto = pd.read_csv('union_lotto.csv')

data = []
for col in lotto.columns:
    values = lotto[col].values.tolist()
    for i in range(0, len(values) - 101):
        temp = [str(v) for v in values[i:i + 100]]
        line = {
            "conversations": [
                {
                    "from": "human",
                    "value": "根据以下数字序列预测下一个数字：{}".format(','.join(temp))
                },
                {
                    "from": "gpt",
                    "value": "下一个数字是：{}".format(values[i + 101])
                }
            ]
        }
        data.append(line)
split = int(len(data) * 0.7)
with open('union_lotto_train.json', 'w') as fw:
    for line in data[:split]:
        fw.write(json.dumps(line, ensure_ascii=False) + '\n')
with open('union_lotto_dev.json', 'w') as fw:
    for line in data[split:]:
        fw.write(json.dumps(line, ensure_ascii=False) + '\n')
