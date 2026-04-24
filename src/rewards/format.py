"""奖励函数：验证模型输出格式。"""

import re

import pandas as pd


def output_format_check(prompt: str, completion: str, example: dict) -> float:
	reward = 0.0
	try:
		# Add synthetic <think> as it's already part of the prompt and prefilled
		# for the assistant to more easily match the regex.
		completion = "<think>" + completion

		regex = (
			r"^<think>\s*([^<]*(?:<(?!/?think>)[^<]*)*)\s*<\/think>\n"
			r"<guess>\s*([\s\S]*?)\s*<\/guess>$"
		)
		match = re.search(regex, completion, re.DOTALL)
		if match is None or len(match.groups()) != 2:
			return 0.0

		guess = match.groups()[1].strip()
		if len(guess) != 5:
			return 0.1

		word_list = pd.read_csv(str(example["word_list"]))
		if guess not in word_list["Word"].values:
			return 0.5

		reward = 1.0
	except Exception:
		pass

	return reward


__all__ = ["output_format_check"]