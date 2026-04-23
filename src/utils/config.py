import os
from dotenv import load_dotenv
from predibase import Predibase

def get_predibase_client() -> Predibase:
    """
    加载环境变量并初始化 Predibase 客户端。
    用于在各个模块中安全、方便地提取云端环境授权。
    """
    # 从本地的 .env 文件加载环境变量
    load_dotenv()
    
    # 获取 Predibase 的 API Key
    api_key = os.environ.get("PREDIBASE_API_KEY")
    if not api_key:
        raise ValueError("未设置 PREDIBASE_API_KEY 环境变量，请检查 .env。")
    
    # 返回建立好的客端实例
    return Predibase(api_token=api_key)
