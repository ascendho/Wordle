from predibase import (
    GRPOConfig,
    RewardFunctionsConfig,
    RewardFunctionsRuntimeConfig,
    SamplingParamsConfig,
)
from src.utils.config import get_predibase_client
from src.data.loader import get_wordle_grpo_dataset

# 引入强化学习的打分环境/奖励函数
from src.rewards.format import output_format_check
from src.rewards.feedback import uses_previous_feedback
from src.rewards.entropy import guess_value


def run_sft_grpo_training(sft_version: str = "wordle/1"):
    """
    SFT + GRPO 联合训练流程。
    从一个先前执行过部分 SFT 微调的网络权重继续启动 GRPO 策略，
    这样网络在早期不会过于盲目瞎猜。

    参数:
        sft_version (str): 选择之前完成 SFT 版本适配器字符串标识，比如 wordle/1
    """
    # 建立与云服务平台连接
    pb = get_predibase_client()

    # 拉取训练 RL 使用的情境数据分布
    dataset = get_wordle_grpo_dataset(pb)
    pb.repos.create(name="wordle", exists_ok=True)

    # 建立奖励及调度配置，相较于纯净起点的 GRPO，这里会加速迭代并调低 epoch 跟 generation 并发
    config = GRPOConfig(
        base_model="qwen2-5-7b-instruct",
        reward_fns=RewardFunctionsConfig(
            runtime=RewardFunctionsRuntimeConfig(packages=["pandas"]),
            functions={
                "output_format_check": output_format_check,
                "uses_previous_feedback": uses_previous_feedback,
                "guess_value": guess_value,
            },
        ),
        epochs=3,
        enable_early_stopping=False,
        sampling_params=SamplingParamsConfig(max_tokens=4096),
        num_generations=8,
    )

    # 建立作业：重点传入 continue_from_version 让底层权重承接之前的结果！
    job = pb.finetuning.jobs.create(
        config=config,
        continue_from_version=sft_version,
        dataset=dataset,
        repo="wordle",
        description="Wordle SFT+GRPO 混合训练任务",
    )
    print(f"基于 {sft_version} 的联合强化训练 SFT+GRPO 任务已触发！ Job ID: {getattr(job, 'id', 'Unknown')}")