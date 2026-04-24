"""Wordle benchmark runner: execute full multi-turn games and summarize solved-rate metrics."""

import pandas as pd
import re

from src.utils.config import get_predibase_client

# Wordle 对局参数（与原版游戏一致）
MAX_TURNS = 6
WORD_LENGTH = 5


def extract_guess(completion: str) -> str:
    """从模型输出中提取 <guess>...</guess> 内的猜词内容。"""
    match = re.search(r"<guess>\s*([\s\S]*?)\s*<\/guess>", completion, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def extract_secret_word(example_row: pd.Series) -> str:
    """跨不同字段名提取真实 secret。"""
    for key in ("secret", "target", "answer", "word", "secret_word"):
        value = example_row.get(key)
        if isinstance(value, str) and len(value.strip()) == WORD_LENGTH:
            return value.strip().upper()
    return ""


def _compute_feedback(secret: str, guess: str) -> str:
    """
    根据 Wordle 规则生成反馈字符串，格式与训练数据保持一致：
      - 完全正确：`X(✓) `
      - 字母存在但位置不对：`X(-) `
      - 字母不在 secret 中：`X(x) `
    """
    secret_chars = list(secret)
    feedback = [None] * WORD_LENGTH

    for i, (g_char, s_char) in enumerate(zip(guess, secret)):
        if g_char == s_char:
            feedback[i] = f"{g_char}(✓) "
            secret_chars[i] = None

    for i, g_char in enumerate(guess):
        if feedback[i] is not None:
            continue
        if g_char in secret_chars:
            feedback[i] = f"{g_char}(-) "
            secret_chars[secret_chars.index(g_char)] = None
        else:
            feedback[i] = f"{g_char}(x) "

    return "".join(feedback).strip()


def _render_history(history: list) -> str:
    """将历史猜测格式化为 prompt 末尾的状态块。"""
    if not history:
        return "(暂无历史猜测)"
    lines = []
    for idx, (past_guess, past_feedback) in enumerate(history, start=1):
        lines.append(f"第 {idx} 次猜测: {past_guess} -> {past_feedback}")
    return "\n".join(lines)


def _build_turn_prompt(base_prompt: str, history: list) -> str:
    """在数据集原始 prompt 末尾追加当前对局状态。"""
    history_block = _render_history(history)
    return (
        f"{base_prompt}\n\n"
        f"[对局状态] 已进行 {len(history)}/{MAX_TURNS} 次猜测。\n"
        f"{history_block}\n\n"
        f"请根据以上反馈给出下一次猜测，严格遵循 <think>...</think>\\n<guess>...</guess> 格式。"
    )


def _call_adapter(adapter, prompt: str, temperature: float, max_new_tokens: int):
    """兼容不同 Predibase SDK 版本的 generate 接口。"""
    # 新版 SDK 支持 temperature；若调用签名较旧，则回退到只传基础生成参数。
    try:
        return adapter.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    except TypeError:
        return adapter.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )


def _play_single_game(
    adapter,
    base_prompt: str,
    secret: str,
    temperature: float,
    max_new_tokens: int,
) -> dict:
    """
    对单个 secret 执行最多 6 轮的完整 Wordle 对局。

    返回结果既包含是否通关，也包含停止原因：
    - `solved`: 在 6 轮内命中答案
    - `missing_guess_turn_k`: 模型没有按约定产出 `<guess>`
    - `malformed_guess_turn_k`: 产出的猜词不是合法五字母字母串
    - `exhausted`: 每轮都合法，但六次机会仍未猜中
    """
    history: list = []
    reason = "exhausted"

    for turn in range(1, MAX_TURNS + 1):
        # 每一轮都把最新历史重新拼回 prompt，模拟真实 Wordle 中“看完反馈再猜下一次”。
        prompt = _build_turn_prompt(base_prompt, history)
        response = _call_adapter(adapter, prompt, temperature, max_new_tokens)
        completion = response.generated_text if hasattr(response, "generated_text") else str(response)

        raw_guess = extract_guess(completion)
        if not raw_guess:
            reason = f"missing_guess_turn_{turn}"
            break

        guess = raw_guess.upper()
        if len(guess) != WORD_LENGTH or not guess.isalpha():
            reason = f"malformed_guess_turn_{turn}"
            break

        feedback = _compute_feedback(secret, guess)
        history.append((guess, feedback))

        if guess == secret:
            return {
                "solved": True,
                "turns": turn,
                "reason": "solved",
                "history": history,
            }

    return {
        "solved": False,
        "turns": len(history) if history else 0,
        "reason": reason,
        "history": history,
    }


def run_benchmark(
    adapter_id: str,
    num_games: int = 10,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
):
    """
    完整 Wordle 对局基准测试器。

    对每个 secret 最多进行 6 轮真实对局：每轮将历史反馈拼回 prompt 让模型继续猜测，
    通关即停；六轮仍未命中则记为失败。默认 temperature=0.7，可通过 CLI 覆盖。
    """
    print(
        f"--- 启动 Wordle 基准测试 ---\n"
        f"    adapter        = {adapter_id}\n"
        f"    num_games      = {num_games}\n"
        f"    temperature    = {temperature}\n"
        f"    max_new_tokens = {max_new_tokens}"
    )
    pb = get_predibase_client()

    try:
        dataset = pb.datasets.get("wordle_grpo_data")
        df = dataset.to_pandas()

        # 数据集中可能包含同一个 secret 的多条样本；benchmark 只保留每个答案一局，
        # 否则成功率会被重复 secret 放大。
        secret_cols = [c for c in ("secret", "target", "answer", "word", "secret_word") if c in df.columns]
        if secret_cols:
            df = df.drop_duplicates(subset=secret_cols[:1])
        df = df.head(num_games)

        try:
            adapter = pb.adapters.get(adapter_id)
        except Exception as e:
            print(f"注意！适配器 {adapter_id} 未就绪或被异常拦截：{e}")
            print("请确认在 Predibase 上已完成模型 Deployment，基准测试流程即将退出。")
            return

        total_games = 0
        solved_count = 0
        guesses_in_solved: list = []
        failure_reasons: dict = {}

        print(f"开始对 {len(df)} 个独立 secret 进行完整对局基准测试...\n")

        for _, row in df.iterrows():
            expected_secret = extract_secret_word(row)
            base_prompt = row.get("prompt", "")

            if not expected_secret or not isinstance(base_prompt, str) or not base_prompt.strip():
                failure_reasons["missing_secret_or_prompt"] = failure_reasons.get("missing_secret_or_prompt", 0) + 1
                continue

            total_games += 1
            result = _play_single_game(
                adapter=adapter,
                base_prompt=base_prompt,
                secret=expected_secret,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

            status = "WIN " if result["solved"] else "LOSS"
            print(f"[{status}] secret={expected_secret} turns={result['turns']} reason={result['reason']}")

            if result["solved"]:
                solved_count += 1
                guesses_in_solved.append(result["turns"])
            else:
                # 失败原因单独计数，便于区分“策略猜不中”与“输出结构出错”等不同问题。
                failure_reasons[result["reason"]] = failure_reasons.get(result["reason"], 0) + 1

        # 平均猜词次数只在获胜局上统计，和 README/配图中的指标口径保持一致。
        avg_guesses = sum(guesses_in_solved) / len(guesses_in_solved) if guesses_in_solved else 0.0
        results = {
            "Model": adapter_id,
            "成功通关数 (Solved Games)": solved_count,
            "总对局数 (Total Games)": total_games,
            "获胜组内的平均猜词次数 (Avg # Guesses)": round(avg_guesses, 2),
        }

        results_df = pd.DataFrame([results])
        print("\n--- 模型基准测试总评 (Benchmark Results) ---")
        print(results_df.to_string(index=False))
        if failure_reasons:
            print("\n失败原因分布：")
            for reason, count in sorted(failure_reasons.items(), key=lambda kv: -kv[1]):
                print(f"  - {reason}: {count}")
        print("------------------------------------\n")
    except Exception as e:
        print(f"基准测试环境引发致命错误阻断了测算: {e}")