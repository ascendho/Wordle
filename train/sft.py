from predibase import SFTConfig
from src.utils.config import get_predibase_client
from src.data.loader import get_wordle_sft_dataset

def run_sft_training():
    """
    执行纯纯的监督微调 (SFT) 流程！
    此逻辑将使用已标记的目标 Wordle SFT 演示数据，令模型初步学会 XML 标签输出格式
    并具有基础词语联想理解能力。
    """
    # 步骤1：初始化环境
    pb = get_predibase_client()
    
    # 步骤2：从系统缓存加载特定的 SFT 数据集
    dataset = get_wordle_sft_dataset(pb)
    
    # 步骤3：确保我们在 Predibase 管理平台上存在 `wordle` 这个实验记录仓库
    pb.repos.create(name="wordle", exists_ok=True)
    
    # 步骤4：装配 LoRA 微调配置，注入模型底座 Qwen 2.5 7B
    config = SFTConfig(
        base_model="qwen2-5-7b-instruct",
        epochs=10, 
        rank=64, # LoRA 参数秩大小
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    )
    
    # 步骤5：下发构建 SFT 的微调 Job 给计算后端资源执行
    job = pb.finetuning.jobs.create(
        config=config,
        dataset=dataset,
        repo="wordle",
        description="Wordle SFT 监督微调任务, 10 轮"
    )
    
    # 获取任务并打印调度ID
    print("SFT 微调任务创建执行成功！ Job ID：", getattr(job, "id", "Unknown"))
