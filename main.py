import argparse

def main():
    """主入口函数：解析命令行参数并调度对应的子模块任务"""
    parser = argparse.ArgumentParser(description="Wordle SFT/GRPO 微调流程")
    
    # 定义要运行的阶段
    parser.add_argument(
        "--run", 
        type=str, 
        choices=["data", "train", "eval"], 
        required=True,
        help="要运行的流水线阶段：准备目标数据, 执行训练, 或运行评估基准。"
    )
    
    # 针对训练阶段，指定训练类型
    parser.add_argument(
        "--type", 
        type=str, 
        choices=["sft", "grpo", "sftgrpo"], 
        default="grpo",
        help="具体的训练执行类型（仅在 --run train 时生效）。可以选择 sft, grpo 或联合的 sftgrpo。"
    )
    
    # 针对继续训练与评估阶段的模型版本标识
    parser.add_argument(
        "--adapter", 
        type=str, 
        default="wordle/1",
        help="要评估或作为基座使用的适配器/模型版本号 (例如 wordle/1)"
    )

    args = parser.parse_args()

    # 1. 数据准备分支
    if args.run == "data":
        print("正在为 Wordle 准备数据集...")
        # 局部导入以避免在不需要时触发不必要的模块加载
        from src.utils.config import get_predibase_client
        from src.data.loader import get_wordle_grpo_dataset, get_wordle_sft_dataset
        
        # 初始化 Predibase 客户端并获取/缓存数据集
        pb = get_predibase_client()
        get_wordle_sft_dataset(pb)
        get_wordle_grpo_dataset(pb)
        print("数据集成功获取并缓存至 Predibase。")

    # 2. 模型训练分支
    elif args.run == "train":
        if args.type == "sft":
            # 运行监督微调 (Supervised Fine-Tuning)
            from train.sft import run_sft_training
            run_sft_training()
            
        elif args.type == "grpo":
            # 运行生成式奖励策略优化 (Generative Reward Policy Optimization)
            from train.grpo import run_grpo_training
            run_grpo_training()
            
        elif args.type == "sftgrpo":
            # 基于已有的 SFT 模型版本继续运行 GRPO
            from train.sftgrpo import run_sft_grpo_training
            run_sft_grpo_training(sft_version=args.adapter)

    # 3. 模型评估分支
    elif args.run == "eval":
        # 运行评估或推理验证，测试模型的破题准确率与平均猜测次数
        from eval.evaluate import run_evaluation
        run_evaluation(adapter_id=args.adapter)

if __name__ == "__main__":
    main()
