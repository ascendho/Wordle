import pandas as pd
import re
import ast

from src.utils.config import get_predibase_client

def extract_guess(completion: str) -> str:
    """辅助手段：从模型返回的串中通过正则表达式抓取 <guess> 和 </guess> 之间的单词猜想内容。"""
    match = re.search(r"<guess>\s*([\s\S]*?)\s*<\/guess>", completion, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def extract_secret_word(example_row: pd.Series) -> str:
    """提取真实参考词汇助手：跨词库可能具有不同的特征字段命名（如secret, word等）。该函数能做健壮匹配。"""
    for key in ("secret", "target", "answer", "word", "secret_word"):
        value = example_row.get(key)
        # 判断是字符且刚好长为5时捕获返回并变大写
        if isinstance(value, str) and len(value.strip()) == 5:
            return value.strip().upper()
    return ""


def safe_history_len(raw_history) -> int:
    """
    计算安全长度器：将表格内提取到的字符串或对象结构化游戏历史 past_guess_history 转义执行，
    借此用来得知模型这是在第几次游戏之内回答的内容。
    """
    if isinstance(raw_history, list):
        return len(raw_history)
    if not isinstance(raw_history, str) or not raw_history.strip():
        return 0
    try:
        parsed = ast.literal_eval(raw_history)
        return len(parsed) if isinstance(parsed, list) else 0
    except Exception:
        return 0

def run_evaluation(adapter_id: str):
    """
    主评估执行逻辑：
    抽入部分的测试集问题用我们的已微调模型或策略接口进行生成推理（Inference）,
    将模型真实给出的输出答卷与底层隐藏真实词对比，
    输出通关率统计结果与平均推理步数。
    """
    print(f"--- 启动基准评估！测试模型适配器: {adapter_id} ---")
    pb = get_predibase_client()
    
    try:
        # 1. 抓取用于展示小闭环测验的参考数据集对象 (取前10个问题用来演示)
        dataset = pb.datasets.get("wordle_grpo_data")
        df = dataset.to_pandas().head(10)
        
        # 2. 我们获取用户部署的终端推理节点 (预测点：要求必须在对应云端平台已发布适配为终端)
        try:
            adapter = pb.adapters.get(adapter_id)
        except Exception as e:
            print(f"注意！适配器 {adapter_id} 没有可用的就绪推理服务或被异常拦截. 详细日志：{e}")
            print("请确认在云提供商中已经完成模型的下发操作 (Deployment)。正在返回跳出...")
            return

        total_games = len(df)
        solved_count = 0
        guesses_in_solved = []

        print(f"开始迭代验证 {total_games} 条不同的 Wordle 问题...")
        
        # 遍历数据集开始测试
        for index, row in df.iterrows():
            prompt = row['prompt']
            expected_secret = extract_secret_word(row)
            
            # 没有基准对比目标的时候只能直接跳过这题
            if not expected_secret:
                continue
            
            # 3. 针对这一条提问对远端 Inference 环境发指令，生成预测文本长篇过程
            response = adapter.generate(
                prompt=prompt,
                max_new_tokens=512,
            )
            
            # 抽离不同 API 可能存在的 payload 对象
            completion = response.generated_text if hasattr(response, 'generated_text') else str(response)
            
            # 使用提取工具拿到最终打分词
            guess = extract_guess(completion)
            
            # 评分与计数策略：
            # 如果猜到的词与标准一致直接记录获胜！
            if guess.upper() == expected_secret:
                solved_count += 1
                # 根据该关游戏历史环境状态算出其花费的总步数
                guesses_in_solved.append(safe_history_len(row.get('past_guess_history')) + 1)
        
        # 计算在所通关的关卡样本里，每一次模型到底要花多少步骤，体现模型质量策略是否够优效率高
        avg_guesses = sum(guesses_in_solved) / len(guesses_in_solved) if guesses_in_solved else 0.0

        # 返回精美图表总结展示
        results = {
            "Model": adapter_id,
            "成功通关数 (Solved Games)": solved_count,
            "总基准用例组 (Total Evaluation Run)": total_games,
            "获胜组内的平均猜词次数 (Avg # Guesses)": round(avg_guesses, 2)
        }
        
        results_df = pd.DataFrame([results])
        print("\n--- 模型基准测验总评 (Evaluation Results) ---")
        print(results_df.to_string(index=False))
        print("------------------------------------\n")
    except Exception as e:
        print(f"评估环境引发致命错误阻断了测算: {e}")
