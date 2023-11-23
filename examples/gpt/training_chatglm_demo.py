# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import argparse
import os
import sys
from datetime import datetime

from loguru import logger

sys.path.append('../..')
from textgen import GptModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='../data/sharegpt_zh_100_format.jsonl', type=str, help='Train file')
    parser.add_argument('--test_file', default='../data/sharegpt_zh_100_format.jsonl', type=str, help='Test file')
    parser.add_argument('--eval_file', default='../data/sharegpt_zh_100_format.jsonl', type=str, help='Test file')
    parser.add_argument('--model_type', default='chatglm', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='THUDM/chatglm-6b', type=str,
                        help='Transformers model or path')
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_predict', action='store_true', help='Whether to run predict.')
    parser.add_argument('--bf16', action='store_true', help='Whether to use bf16 mixed precision training.')
    parser.add_argument('--output_dir', default='./outputs-chatglm-demo/', type=str, help='Model output directory')
    parser.add_argument('--prompt_template_name', default='vicuna', type=str, help='Prompt template name')
    parser.add_argument('--max_seq_length', default=512, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=512, type=int, help='Output max sequence length')
    parser.add_argument('--num_epochs', default=0.2, type=float, help='Number of training epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--eval_steps', default=50, type=int, help='Eval every X steps')
    parser.add_argument('--save_steps', default=50, type=int, help='Save checkpoint every X steps')
    parser.add_argument('--wandb_project', default="Textgen", type=str, help='wandb project')
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
    logger.info(args)
    model = None
    # fine-tune Llama model
    if args.do_train:
        logger.info('Loading data...')
        if args.wandb_project:
            os.environ['WANDB_PROJECT'] = args.wandb_project
        model_args = {
            "use_peft": True,
            "overwrite_output_dir": True,
            "reprocess_input_data": True,
            "max_seq_length": args.max_seq_length,
            "max_length": args.max_length,
            "per_device_train_batch_size": args.batch_size,
            "eval_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "output_dir": args.output_dir,
            "resume_from_checkpoint": args.output_dir,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "bf16": args.bf16,
            "prompt_template_name": args.prompt_template_name,
            "wandb_project": args.wandb_project,
            "report_to": args.report_to,
            "early_stopping_consider_epochs": args.early_stopping_consider_epochs,
            "early_stopping_delta": args.early_stopping_delta,
            "early_stopping_metric": args.early_stopping_metric,
            "early_stopping_metric_minimize": args.early_stopping_metric_minimize,
            "early_stopping_patience": args.early_stopping_patience,
            "use_early_stopping": args.use_early_stopping,
            "run_name": "chatglm-{}".format(datetime.now().strftime("%m%d%H%M%S"))
        }
        model = GptModel(args.model_type, args.model_name, args=model_args)
        model.train_model(args.train_file, eval_data=args.eval_file)
    if args.do_predict:
        if model is None:
            model = GptModel(
                args.model_type, args.model_name,
                peft_name=args.output_dir,
                args={'use_peft': True, 'eval_batch_size': args.batch_size, "max_length": args.max_length, }
            )

        # response = model.predict(["介绍下北京", "介绍下南京", "给出5个必去武汉的理由"])
        # print(response)

        # Chat model with multi turns conversation
        response, history = model.chat(
            '根据以下数字序列预测下一个数字：22,27,24,27,16,32,14,23,13,23,26,27,25,27,9,19,18,27,30,30,19,22,28,25,27,18,18,10,29,22,25,28,25,23,14,23,28,21,29,26,30,23,26,23,24,19,18,19,28,24,28,21,12,20,24,25,11,32,25,32,29,21,24,21,30,30,11,16,23,21,21,22,24,20,30,25,21,32,25,17,21,21,31,25,29,18,21,32,24,30,32,27,24,28,18,28,26,30,17,19')
        print(response)


if __name__ == '__main__':
    main()
