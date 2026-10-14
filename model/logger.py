#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志模块
"""
import logging
from rich.logging import RichHandler

def get_logger(experiment_name, log_file="train.log"):
    """创建带有实验名称前缀的日志记录器"""
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    # 清除已存在的处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(f"%(asctime)s | {experiment_name} | %(levelname)s | %(message)s")
    console_handler.setFormatter(console_formatter)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(f"%(asctime)s | {experiment_name} | %(levelname)s | %(message)s")
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
