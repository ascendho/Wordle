"""奖励函数：验证模型输出是否满足训练时约定的 XML 结构与猜词约束。"""

import re

import pandas as pd


def output_format_check(prompt: str, completion: str, example: dict) -> float:
	"""
	按“结构完整性 -> 词长正确 -> 词表合法”三层约束返回分段奖励。

	评分规则故意做成逐级递增，而不是只有 0/1：
	1. 完全无法解析出 `<think>` / `<guess>` 结构时返回 0.0。
	2. 结构能解析，但猜词长度不是 5 时返回 0.1，表示“格式接近正确但还不能玩游戏”。
	3. 长度正确，但猜词不在合法词表中时返回 0.5，表示“已经像一个合法回合，只是猜了不存在的词”。
	4. 同时满足结构、长度和词表校验时返回 1.0。

	prompt 参数由 Predibase 奖励函数接口统一传入，这里不直接使用；
	example 只依赖其中的 `word_list` 路径来校验猜词是否合法。
	"""
	reward = 0.0
	try:
		# 训练/采样时 `<think>` 起始标签通常已经由 prompt 预填，因此 completion
		# 实际上是从 think 内容开始。这里补一个 synthetic `<think>`，让正则检查
		# 面向“完整最终输出”而不是面向“被截断的 completion 片段”。
		completion = "<think>" + completion

		# 该正则要求输出严格是两段结构：一个 think 块，下一行紧接一个 guess 块。
		# 其中 think 块内部允许普通文本和非 think 标签字符，但不允许再次嵌套 think。
		regex = (
			r"^<think>\s*([^<]*(?:<(?!/?think>)[^<]*)*)\s*<\/think>\n"
			r"<guess>\s*([\s\S]*?)\s*<\/guess>$"
		)
		match = re.search(regex, completion, re.DOTALL)
		if match is None or len(match.groups()) != 2:
			return 0.0

		guess = match.groups()[1].strip()
		# 长度错误说明输出形式已经可解析，但还不能被 Wordle 环境直接消费。
		if len(guess) != 5:
			return 0.1

		# 只有落在词表中的五字母词才被视为真正可执行的猜测。
		word_list = pd.read_csv(str(example["word_list"]))
		if guess not in word_list["Word"].values:
			return 0.5

		reward = 1.0
	except Exception:
		# 奖励函数必须尽量“失败即保守”，不能因为解析异常把一次坏样本误判成高奖励。
		pass

	return reward


__all__ = ["output_format_check"]