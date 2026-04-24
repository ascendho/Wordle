from predibase import (
    GRPOConfig,
    RewardFunctionsConfig,
    RewardFunctionsRuntimeConfig,
    SamplingParamsConfig,
)
from src.utils.config import get_predibase_client
from src.data.loader import get_wordle_grpo_dataset

# 局部引入我们的核心打分环境及奖励函数机制
from src.rewards.format import output_format_check
from src.rewards.feedback import uses_previous_feedback
from src.rewards.entropy import guess_value

def run_grpo_training():
    """
    运行纯净的 GRPO 生成型奖励策略优化训练。
    不同于无脑学习示范数据，这里我们会让模型在执行过程中通过我们的定义奖励反馈
    自己总结出打游戏的最佳策略。
    """
    pb = get_predibase_client()
    
    # 获取用于 GRPO 的探索数据集
    dataset = get_wordle_grpo_dataset(pb)
    
    # 确保云空间有实验跟踪库
    pb.repos.create(name="wordle", exists_ok=True)
    
    # 装配 GRPO 参数环境，绑定相关的奖励函数
    config = GRPOConfig(
        base_model="qwen2-5-7b-instruct",
        reward_fns=RewardFunctionsConfig(
            runtime=RewardFunctionsRuntimeConfig(packages=["pandas"]),
            functions={
                "output_format_check": output_format_check,        # 检查指令格式是否严格服从规范
                "uses_previous_feedback": uses_previous_feedback,  # 检查是否善用了前一步给予的线索
                "guess_value": guess_value,                        # 验证猜测熵信息效率增益
            }
        ),

        # 控制每次采样大小，避免陷入冗长思考链
        sampling_params=SamplingParamsConfig(max_tokens=4096),

        # GRPO 的核心特征。
        # 对于每一个游戏状态，模型会一口气生成 8 种不同的思考过程和猜测结果。
        # GRPO 会在这 8 个结果中进行内部“选优”，从而优化模型参数。
        num_generations=8 
    )
    
    # 将包含奖励系统的强化微调触发投递出去
    job = pb.finetuning.jobs.create(
        config=config,
        dataset=dataset,
        repo="wordle",
        description="Wordle GRPO 强化学习任务"
    )
    print("GRPO (强化策略优化) 更新任务调度成功！ Job ID:", getattr(job, "id", "Unknown"))
