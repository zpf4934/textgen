# -*- coding:utf-8 -*-
"""
# File       : lottery_ticket.py
# Time       ：2023/9/6 14:19
# Author     ：andy
# version    ：python 3.9
"""
import torch

torch.multiprocessing.set_start_method('spawn')
import argparse
import json
import os.path
import sys
import re
import random
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from loguru import logger

sys.path.append('../')
from textgen import GptModel


class Lotto:
    def __init__(self, save_path="union_lotto.csv", step=1000, split="6:2:2", url=None):
        self.url = url
        self.save_path = save_path
        self.pages = []
        self.ball = []
        self.history = None
        self.step = step
        self.split = split

    def check_exits(self):
        date = []
        if os.path.exists(self.save_path):
            self.history = pd.read_csv(self.save_path, dtype=str)
            date = self.history['date'].values.tolist()
        return date

    def urls(self):
        date = self.check_exits()
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, 'html.parser')
        page_list = soup.select('div.iSelectList a')
        for p in page_list[::-1]:
            if p.string not in date:
                self.pages.append(p['href'])
        logger.info("即将抓取{}期".format(len(self.pages)))

    def split_data(self, data):
        splits = self.split.split(':')
        splits = [int(s) for s in splits]
        train_num = int(len(data) * splits[0] / sum(splits))
        eval_num = int(len(data) * splits[1] / sum(splits))
        with open(os.path.splitext(os.path.basename(self.save_path))[0] + '_train.json', 'w') as fw:
            for line in data[:train_num]:
                fw.write(json.dumps(line, ensure_ascii=False) + '\n')
        with open(os.path.splitext(os.path.basename(self.save_path))[0] + '_dev.json', 'w') as fw:
            for line in data[train_num:train_num + eval_num]:
                fw.write(json.dumps(line, ensure_ascii=False) + '\n')
        with open(os.path.splitext(os.path.basename(self.save_path))[0] + '_test.json', 'w') as fw:
            for line in data[train_num + eval_num:]:
                fw.write(json.dumps(line, ensure_ascii=False) + '\n')
        logger.info("数据格式化完成！")

    def write_file(self):
        df = pd.DataFrame(self.ball)
        if self.history is not None:
            df = pd.concat([self.history, df])
        df.sort_values(by='date', inplace=True)
        df.to_csv(self.save_path, index=False)


class UnionLotto(Lotto):
    def __init__(self, save_path="union_lotto.csv", step=300, split="6:2:2"):
        url = "http://kaijiang.500.com/ssq.shtml"
        super().__init__(save_path, step, split, url)

    def get_data(self):
        for page in tqdm(self.pages):
            try:
                html = requests.get(page, timeout=(3, 5)).text
                soup = BeautifulSoup(html, 'html.parser')
                box = soup.select('div.ball_box01 ul li')
                date = soup.select('font.cfont2 strong')
                self.ball.append(
                    {'date': date[0].string, 'red1': box[0].string, 'red2': box[1].string, 'red3': box[2].string,
                     'red4': box[3].string, 'red5': box[4].string, 'red6': box[5].string, 'blue': box[6].string})
            except Exception as e:
                logger.error(e)
                logger.error("error page {}".format(page))

    def prepare(self):
        logger.info("数据格式化中。。。。")
        lotto = pd.read_csv(self.save_path, index_col='date', dtype=str)
        data = []
        for i in range(0, len(lotto) - self.step - 1):
            exclude = []
            for col in lotto.columns:
                values = lotto[col].values.tolist()
                temp = [str(v) for v in values[i:i + self.step]]
                if "red" in col:
                    prompt = "根据以下数字序列：{}。预测下一个数字，范围1到33。"
                    if exclude:
                        prompt += "{}除外。".format(",".join(exclude))
                else:
                    prompt = "根据以下数字序列：{}。预测下一个数字，范围1到16。"
                exclude.append(str(values[i + self.step + 1]))
                line = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": prompt.format(','.join(temp))
                        },
                        {
                            "from": "gpt",
                            "value": "下一个数字是：{}".format(values[i + self.step + 1])
                        }
                    ]
                }
                data.append(line)
        random.shuffle(data)
        self.split_data(data)

    def get_lasted(self):
        logger.info("最新号码：")
        lotto = pd.read_csv(self.save_path, index_col='date', dtype=str)
        lasted = []
        for col in lotto.columns:
            values = lotto[col].values.tolist()
            if "red" in col:
                logger.info("根据以下数字序列：{}。预测下一个数字，范围1到33。".format(','.join(values[-self.step:])))
                lasted.append("根据以下数字序列：{}。预测下一个数字，范围1到33。".format(','.join(values[-self.step:])))
            else:
                logger.info("根据以下数字序列：{}。预测下一个数字，范围1到16。".format(','.join(values[-self.step:])))
                lasted.append("根据以下数字序列：{}。预测下一个数字，范围1到16。".format(','.join(values[-self.step:])))
        return lasted

    def download(self):
        self.urls()
        self.get_data()
        self.write_file()
        logger.info("数据抓取完成！")

    def auto_predict(self, model):
        lasted = self.get_lasted()
        exclude = []
        red_num = 0
        blue_ball = None
        for l in lasted:
            if '范围1到33' in l:
                red_num += 1
            if red_num == 1 or '范围1到16' in l:
                remove = False
            else:
                remove = True
            if remove:
                e = input("输入需要排除的数字,默认为{}：".format(','.join(exclude)))
                if not e:
                    e = ','.join(exclude)
                l += e
            logger.info(l)
            response = model.predict(l)
            logger.info(response)
            balls = re.findall(r"\d+", response)
            if balls:
                if '范围1到33' in l:
                    exclude.append(balls[-1])
                else:
                    blue_ball = balls[-1]
        if blue_ball:
            exclude.append(blue_ball)
        if len(exclude) != 7:
            logger.warning("预测结果异常！！！！")
        logger.info("预测结果为：{}".format(','.join(exclude)))
        exclude = [int(i) for i in exclude]
        exclude_red = exclude[:-1]
        exclude_blue = [exclude[-1]]
        for _ in range(4):
            red = [ball for ball in range(1, 34) if ball not in exclude_red]
            blue = [ball for ball in range(1,17) if ball not in exclude_blue]
            random_red = sorted(random.sample(red, k=6))
            random_blue = sorted(random.sample(blue, k=1))
            exclude_red = exclude_red + random_red
            exclude_blue = exclude_blue + random_blue
            logger.info("排除预测随机结果为：{}".format(','.join([str(i) for i in random_red + random_blue])))

    def analyze(self):
        df = pd.read_csv(self.save_path)
        red = pd.DataFrame(df[['red1', 'red2', 'red3', 'red4', 'red5', 'red6']].values.reshape(-1, 1)).value_counts(
            normalize=True)
        print("red frequencies : {}".format(red))
        blue = pd.DataFrame(df[['blue']].values.reshape(-1, 1)).value_counts(normalize=True)
        print("blue frequencies :\n {}".format(blue))


class ThreeD(Lotto):
    def __init__(self, save_path="3D.csv", step=300, split="6:2:2"):
        url = "https://kaijiang.500.com/sd.shtml"
        super().__init__(save_path, step, split, url)

    def get_data(self):
        for page in tqdm(self.pages):
            try:
                html = requests.get(page, timeout=(3, 5)).text
                soup = BeautifulSoup(html, 'html.parser')
                box = soup.select('div.ball_box01 ul li')
                date = soup.select('font.cfont2 strong')
                self.ball.append(
                    {'date': date[0].string, 'ball1': box[0].string, 'ball2': box[1].string, 'ball3': box[2].string})
            except Exception as e:
                logger.error(e)
                logger.error("error page {}".format(page))

    def prepare(self):
        logger.info("数据格式化中。。。。")
        lotto = pd.read_csv(self.save_path, index_col='date', dtype=str)
        data = []
        for i in range(0, len(lotto) - self.step - 1):
            for col in lotto.columns:
                values = lotto[col].values.tolist()
                temp = [str(v) for v in values[i:i + self.step]]
                prompt = "根据以下数字序列：{}。预测下一个数字，范围0到9。"
                line = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": prompt.format(','.join(temp))
                        },
                        {
                            "from": "gpt",
                            "value": "下一个数字是：{}".format(values[i + self.step + 1])
                        }
                    ]
                }
                data.append(line)
        random.shuffle(data)
        self.split_data(data)

    def get_lasted(self):
        logger.info("最新号码：")
        lotto = pd.read_csv(self.save_path, index_col='date', dtype=str)
        lasted = []
        for col in lotto.columns:
            values = lotto[col].values.tolist()
            logger.info("根据以下数字序列：{}。预测下一个数字，范围0到9。".format(','.join(values[-self.step:])))
            lasted.append("根据以下数字序列：{}。预测下一个数字，范围0到9。".format(','.join(values[-self.step:])))
        return lasted

    def download(self):
        self.urls()
        self.get_data()
        self.write_file()
        logger.info("数据抓取完成！")

    def auto_predict(self, model):
        lasted = self.get_lasted()
        exclude = []
        for l in lasted:
            logger.info(l)
            response = model.predict(l)
            logger.info(response)
            balls = re.findall(r"\d+", response)
            if balls:
                exclude.append(balls[-1])
        if len(exclude) != 3:
            logger.warning("预测结果异常！！！！")
        logger.info("预测结果为：{}".format(','.join(exclude)))

    def analyze(self):
        df = pd.read_csv(self.save_path)
        red = pd.DataFrame(df[['ball1', 'ball2', 'ball3']].values.reshape(-1, 1)).value_counts(normalize=True)
        print("ball frequencies :\n {}".format(red))


class GrandLotto(Lotto):
    def __init__(self, save_path="grand_lotto.csv", step=300, split="6:2:2"):
        url = "https://kaijiang.500.com/dlt.shtml"
        super().__init__(save_path, step, split, url)

    def get_data(self):
        for page in tqdm(self.pages):
            try:
                html = requests.get(page, timeout=(3, 5)).text
                soup = BeautifulSoup(html, 'html.parser')
                box = soup.select('div.ball_box01 ul li')
                date = soup.select('font.cfont2 strong')
                self.ball.append(
                    {'date': date[0].string, 'red1': box[0].string, 'red2': box[1].string, 'red3': box[2].string,
                     'red4': box[3].string, 'red5': box[4].string, 'blue1': box[5].string, 'blue2': box[6].string})
            except Exception as e:
                logger.error(e)
                logger.error("error page {}".format(page))

    def prepare(self):
        logger.info("数据格式化中。。。。")
        lotto = pd.read_csv(self.save_path, index_col='date', dtype=str)
        data = []
        for i in range(0, len(lotto) - self.step - 1):
            red_exclude = []
            blue_exclude = []
            for col in lotto.columns:
                values = lotto[col].values.tolist()
                temp = [str(v) for v in values[i:i + self.step]]
                if "red" in col:
                    prompt = "根据以下数字序列：{}。预测下一个数字，范围1到35。"
                    if red_exclude:
                        prompt += "{}除外。".format(",".join(red_exclude))
                    red_exclude.append(str(values[i + self.step + 1]))
                else:
                    prompt = "根据以下数字序列：{}。预测下一个数字，范围1到12。"
                    if blue_exclude:
                        prompt += "{}除外。".format(",".join(blue_exclude))
                    blue_exclude.append(str(values[i + self.step + 1]))
                line = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": prompt.format(','.join(temp))
                        },
                        {
                            "from": "gpt",
                            "value": "下一个数字是：{}".format(values[i + self.step + 1])
                        }
                    ]
                }
                data.append(line)
        random.shuffle(data)
        self.split_data(data)

    def get_lasted(self):
        logger.info("最新号码：")
        lotto = pd.read_csv(self.save_path, index_col='date', dtype=str)
        lasted = []
        for col in lotto.columns:
            values = lotto[col].values.tolist()
            if "red" in col:
                logger.info("根据以下数字序列：{}。预测下一个数字，范围1到35。".format(','.join(values[-self.step:])))
                lasted.append("根据以下数字序列：{}。预测下一个数字，范围1到35。".format(','.join(values[-self.step:])))
            else:
                logger.info("根据以下数字序列：{}。预测下一个数字，范围1到12。".format(','.join(values[-self.step:])))
                lasted.append("根据以下数字序列：{}。预测下一个数字，范围1到12。".format(','.join(values[-self.step:])))
        return lasted

    def download(self):
        self.urls()
        self.get_data()
        self.write_file()
        logger.info("数据抓取完成！")

    def auto_predict(self, model):
        lasted = self.get_lasted()
        red_exclude = []
        blue_exclude = []
        red_num = 0
        blue_num = 0
        for l in lasted:
            if '范围1到35' in l:
                red_num += 1
            elif '范围1到12' in l:
                blue_num += 1
            if '范围1到35' in l and red_num > 1:
                red_remove = True
                blue_remove = False
            elif '范围1到12' in l and blue_num > 1:
                red_remove = False
                blue_remove = True
            else:
                red_remove = False
                blue_remove = False
            if red_remove or blue_remove:
                exclude = red_exclude if red_remove else blue_exclude
                e = input("输入需要排除的数字,默认为{}：".format(','.join(exclude)))
                if not e:
                    e = ','.join(exclude)
                l += e
            logger.info(l)
            response = model.predict(l)
            logger.info(response)
            balls = re.findall(r"\d+", response)
            if balls:
                if '范围1到35' in l:
                    red_exclude.append(balls[-1])
                elif '范围1到12' in l:
                    blue_exclude.append(balls[-1])
        if len(red_exclude + blue_exclude) != 7:
            logger.warning("预测结果异常！！！！")
        logger.info("预测结果为：{}".format(','.join(red_exclude + blue_exclude)))

    def analyze(self):
        df = pd.read_csv(self.save_path)
        red = pd.DataFrame(df[['red1', 'red2', 'red3', 'red4', 'red5']].values.reshape(-1, 1)).value_counts(
            normalize=True)
        print("red frequencies : {}".format(red))
        blue = pd.DataFrame(df[['blue1', 'blue2']].values.reshape(-1, 1)).value_counts(normalize=True)
        print("blue frequencies :\n {}".format(blue))


class Model:
    def __init__(self, args):
        self.args = args
        self.model = None

    def train(self):
        logger.info('Loading data...')
        if self.args.wandb_project:
            os.environ['WANDB_PROJECT'] = self.args.wandb_project
        model_args = {
            "use_peft": True,
            "overwrite_output_dir": True,
            "reprocess_input_data": True,
            "max_seq_length": self.args.max_seq_length,
            "max_length": self.args.max_length,
            "per_device_train_batch_size": self.args.batch_size,
            "eval_batch_size": self.args.batch_size,
            "num_train_epochs": self.args.num_epochs,
            "output_dir": self.args.output_dir,
            "resume_from_checkpoint": self.args.output_dir,
            "evaluation_strategy": self.args.evaluation_strategy,
            "save_strategy": self.args.save_strategy,
            "eval_steps": self.args.eval_steps,
            "save_steps": self.args.save_steps,
            "bf16": self.args.bf16,
            "prompt_template_name": self.args.prompt_template_name,
            "wandb_project": self.args.wandb_project,
            "report_to": self.args.report_to,
            "early_stopping_consider_epochs": self.args.early_stopping_consider_epochs,
            "early_stopping_delta": self.args.early_stopping_delta,
            "early_stopping_metric": self.args.early_stopping_metric,
            "early_stopping_metric_minimize": self.args.early_stopping_metric_minimize,
            "early_stopping_patience": self.args.early_stopping_patience,
            "use_early_stopping": self.args.use_early_stopping,
            "run_name": "{}-{}".format(self.args.model_type, datetime.now().strftime("%m%d%H%M%S"))
        }
        self.model = GptModel(self.args.model_type, self.args.model_name, args=model_args)
        self.model.train_model(self.args.train_file, eval_data=self.args.eval_file,
                               compute_metrics=self.compute_metrics)

    def predict(self, query):
        args = json.load(open(os.path.join(self.args.output_dir, "model_args.json")))
        if self.model is None:
            self.model = GptModel(
                args.get('model_type'), args.get('model_name'),
                peft_name=args.get('output_dir'),
                args={'use_peft': args.get('use_peft'), 'eval_batch_size': args.get('eval_batch_size'),
                      "max_length": args.get('max_length')}
            )

        # Chat model with multi turns conversation
        response, history = self.model.chat(query, prompt_template_name=args.get('prompt_template_name'))
        return response

    def compute_metrics(self, pred):
        if not isinstance(pred.label_ids, np.ndarray) or not isinstance(pred.predictions, np.ndarray):
            return {}
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        decoded_preds = self.model.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.model.tokenizer.pad_token_id)
        decoded_labels = self.model.tokenizer.batch_decode(labels, skip_special_tokens=True)
        labels_num = []
        preds_num = []
        for i in range(len(decoded_labels)):
            balls = re.findall(r"\d+", decoded_labels[i])
            if balls:
                labels_num.append(balls[-1])
            else:
                labels_num.append('')
            balls = re.findall(r"\d+", decoded_preds[i])
            if balls:
                preds_num.append(balls[-1])
            else:
                preds_num.append('')
        precision, recall, f1, _ = precision_recall_fscore_support(np.array(labels_num), np.array(preds_num),
                                                                   average='macro')
        score_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        return score_dict


def spider(channel, step):
    if channel == '1':
        channel = UnionLotto(step=step)
        channel.download()
        channel.prepare()
    elif channel == '2':
        channel = ThreeD(step=step)
        channel.download()
        channel.prepare()
    elif channel == '3':
        channel = GrandLotto(step=step)
        channel.download()
        channel.prepare()
    else:
        logger.warning("请选择正确的类别")
        sys.exit(1)


def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default=None, type=str, help='Train file')
    parser.add_argument('--test_file', default=None, type=str, help='Test file')
    parser.add_argument('--eval_file', default=None, type=str, help='Test file')
    parser.add_argument('--model_type', default=None, type=str, choices=['chatglm', 'llama'],
                        help='Transformers model type')
    parser.add_argument('--model_name', default=None, type=str,
                        help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--bf16', action='store_true', help='Whether to use bf16 mixed precision training.')
    parser.add_argument('--output_dir', default=None, type=str, help='Model output directory')
    parser.add_argument('--prompt_template_name', default=None, type=str, choices=['chatglm2', 'llama2-zh'],
                        help='Prompt template name')
    parser.add_argument('--max_seq_length', default=700, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=64, type=int, help='Output max sequence length')
    parser.add_argument('--num_epochs', default=50, type=float, help='Number of training epochs')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--evaluation_strategy', default="epoch", type=str, choices=['no', 'steps', 'epoch'],
                        help='The evaluation strategy to adopt during training')
    parser.add_argument('--save_strategy', default="epoch", type=str, choices=['no', 'steps', 'epoch'],
                        help='The checkpoint save strategy to adopt during training')
    parser.add_argument('--eval_steps', default=200, type=int, help='Eval every X steps')
    parser.add_argument('--save_steps', default=200, type=int, help='Save checkpoint every X steps')
    parser.add_argument('--wandb_project', default=None, type=str, help='wandb project')
    parser.add_argument('--report_to', default="wandb", type=str, help='report')
    parser.add_argument('--early_stopping_consider_epochs', default=True, type=bool,
                        help='early stopping consider epochs')
    parser.add_argument('--early_stopping_delta', default=0.01, type=float, help='early stopping delta')
    parser.add_argument('--early_stopping_metric', default="eval_loss", type=str, help='early stopping metric')
    parser.add_argument('--early_stopping_metric_minimize', default=True, type=bool,
                        help='early stopping metric minimize')
    parser.add_argument('--early_stopping_patience', default=5, type=int, help='early stopping patience')
    parser.add_argument('--use_early_stopping', default=True, type=bool, help='use early stopping')
    args = parser.parse_args()
    return args


def model(lotto=None, mode=None, step=300):
    args = add_parser()
    if lotto == '1':
        lotto = UnionLotto(step=step)
    elif lotto == '2':
        lotto = ThreeD(step=step)
    elif lotto == '3':
        lotto = GrandLotto(step=step)
    if args.train_file is None:
        args.train_file = os.path.splitext(os.path.basename(lotto.save_path))[0] + '_train.json'
    if args.test_file is None:
        args.test_file = os.path.splitext(os.path.basename(lotto.save_path))[0] + '_test.json'
    if args.eval_file is None:
        args.eval_file = os.path.splitext(os.path.basename(lotto.save_path))[0] + '_dev.json'
    method = input("选择要使用的算法：\n1:llama\n2:chatglm\n>")
    if method == "1":
        args.model_type = 'llama'
        args.model_name = '/aigc/modelclub/chinese-alpaca-2-7b'
        args.output_dir = 'outputs_llama2_{}'.format(type(lotto).__name__)
        args.prompt_template_name = 'llama2-zh'
        args.wandb_project = 'textgen_llama'
    elif method == "2":
        args.model_type = 'chatglm'
        args.model_name = '/aigc/modelclub/chatglm2-6b'
        args.output_dir = 'outputs_chatglm2_{}'.format(type(lotto).__name__)
        args.prompt_template_name = 'chatglm2'
        args.wandb_project = 'textgen_chatglm'
    else:
        logger.warning("暂不支持其他算法")
        sys.exit(1)
    logger.info(args)
    m = Model(args)
    if args.do_train or mode == '2':
        m.train()
    if args.do_predict or mode == '3':
        query_mode = input("选择预测方式：\n1:人工输入\n2:自动预测\n>")
        if query_mode == '1':
            while True:
                query = input("按q退出，输入：")
                if query == 'q' or not query:
                    sys.exit(1)
                response = m.predict(query)
                logger.info(response)
        else:
            lotto.auto_predict(m)


def main():
    step = 400
    channel = input(
        "选择操作的彩票：\n1:双色球，每周二、四、日21:15开奖\n2:3D，每天21:15开奖\n3:大乐透，每周一、三、六20:30开奖\n>")
    if channel not in ('1', '2', '3'):
        logger.warning("请选择正确的类别")
        sys.exit(1)

    mode = input("选择模式：\n1:数据抓取\n2:训练\n3:预测\n4:分析\n>")
    if mode == "1":
        spider(channel, step)
    elif mode == "2" or mode == "3":
        spider(channel, step)
        model(channel, mode, step)
    elif mode == "4":
        spider(channel, step)
        if channel == '1':
            lotto = UnionLotto()
        elif channel == '2':
            lotto = ThreeD()
        elif channel == '3':
            lotto = GrandLotto()
        lotto.analyze()
    else:
        logger.warning("请选择正确模式")


if __name__ == '__main__':
    main()
