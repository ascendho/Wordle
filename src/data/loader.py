from datasets import load_dataset
import pandas as pd
from predibase import Predibase

def upload_dataset(pb: Predibase, hf_repo: str, pb_name: str) -> pd.DataFrame:
    """
    核心工具函数：从 HuggingFace 加载数据集，并上传/获取到 Predibase。
    
    参数:
        pb (Predibase): Predibase 客户端实例。
        hf_repo (str): HuggingFace 数据集仓库名称。
        pb_name (str): 在 Predibase 环境中注册的数据集名称。
        
    返回:
        Predibase 的 DataFrame 格式数据集。
    """
    # 获取 HF 数据集的训练集 split 并转换为 Pandas DataFrame
    dataset = load_dataset(hf_repo, split="train")
    df = dataset.to_pandas()
    
    try:
        # 尝试使用名称创建新的 Predibase 数据集
        pb_dataset = pb.datasets.from_pandas_dataframe(df, name=pb_name)
    except Exception:
        # 如果数据集名称已经被使用（抛出异常），则直接通过名称获取该数据集引用
        pb_dataset = pb.datasets.get(pb_name)
        
    return pb_dataset

def get_wordle_grpo_dataset(pb: Predibase):
    """获取/上传用于 GRPO 训练流程的 Wordle 数据集"""
    return upload_dataset(pb, "predibase/wordle-grpo", "wordle_grpo_data")

def get_wordle_sft_dataset(pb: Predibase):
    """获取/上传用于 SFT (监督微调) 流程的 Wordle 数据集"""
    return upload_dataset(pb, "predibase/wordle-sft", "wordle_sft_data")
