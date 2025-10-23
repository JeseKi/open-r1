# -*- coding: utf-8 -*-
"""
该文件用于对数据进行预处理，以便在训练时直接加载处理好的数据。

公开接口:
    main: 主函数，用于执行数据预处理。

使用方法:
    python scripts/preprocess_data.py --config recipes/Qwen2.5-7B-Instruct/sft/config_kaggle_4bit.yaml --output_dir ./processed-data/OpenR1-Math-220k-tokenized
"""
import argparse
import logging
import sys
from pathlib import Path

import datasets
import transformers
import yaml
from transformers.hf_argparser import HfArgumentParser
from transformers.trainer_utils import set_seed

from open_r1.configs import ScriptArguments, SFTConfig
from open_r1.utils import get_dataset, get_tokenizer
from trl import ModelConfig


logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    """
    主函数，用于执行数据预处理。
    """
    set_seed(training_args.seed)

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")

    # 加载数据集和分词器
    dataset = get_dataset(script_args)
    tokenizer = get_tokenizer(model_args, training_args)

    def tokenize_function(examples):
        # 使用 tokenizer 的 chat_template 来格式化消息并进行分词
        # 假设数据集中包含一个名为 'messages' 的列，其中包含聊天对话
        formatted_texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) for messages in examples["messages"]]
        return tokenizer(
            formatted_texts, # type: ignore
            truncation=True,
            max_length=training_args.max_length,
            padding=False,
        )

    # 对数据集进行分词
    logger.info("Tokenizing dataset...")
    # 获取数据集中第一个 split 的列名，并用其移除所有旧列
    column_names = next(iter(dataset.values())).column_names
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=script_args.dataset_num_proc,
        remove_columns=column_names,
    )

    # 保存处理好的数据集
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving tokenized dataset to {output_dir}")
    tokenized_dataset.save_to_disk(str(output_dir))

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Preprocess data for SFT training.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the preprocessed dataset.",
    )
    args = parser.parse_args()

    # 加载 YAML 配置文件
    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)

    # 使用 HfArgumentParser 从字典中解析 dataclass 实例
    # 注意：这里我们假设 config_data 包含了所有 dataclass 所需的参数
    # HfArgumentParser 会自动忽略不需要的参数
    script_parser = HfArgumentParser(ScriptArguments) # type: ignore
    script_args = script_parser.parse_dict(config_data, allow_extra_keys=True)[0]

    model_parser = HfArgumentParser(ModelConfig) # type: ignore
    model_args = model_parser.parse_dict(config_data, allow_extra_keys=True)[0]

    sft_parser = HfArgumentParser(SFTConfig) # type: ignore
    training_args = sft_parser.parse_dict(config_data, allow_extra_keys=True)[0]

    # 命令行指定的 output_dir 优先级更高
    training_args.output_dir = args.output_dir

    main(script_args, training_args, model_args)