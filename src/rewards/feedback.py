"""奖励函数：评估模型是否把之前回合的 Wordle 反馈真正用到了下一次猜测里。"""

import ast
import re


def uses_previous_feedback(prompt: str, completion: str, example: dict) -> float:
	"""
	根据历史反馈对新猜词进行启发式奖励。

	这个奖励不追求“是否已经猜中 secret”，而是专门衡量模型有没有遵守基本策略：
	1. 绿色字母应该尽量保留在已确认正确的位置。
	2. 黄色字母说明“字母存在但当前位置错误”，后续应继续使用，但要换位置。
	3. 灰色字母通常应该被淘汰，继续复用会受到更强惩罚。
	4. 对既不违反约束、又能带来探索的新字母给一个很小的正奖励。

	返回值是这些局部策略分的总和，数值可以大于 1，也可以因为多次违背反馈而变成负数。
	"""
	reward = 0.0
	try:
		# completion 往往从 think 内容开始，这里补上前缀，保持和其它奖励函数一致。
		completion = "<think>" + completion

		regex = r"<guess>\s*([\s\S]*?)\s*<\/guess>$"
		match = re.search(regex, completion, re.DOTALL)
		if match is None or len(match.groups()) != 1:
			return 0.0

		guess = match.groups()[0].strip()
		if len(guess) != 5:
			return 0.0

		past_guess_history = ast.literal_eval(example["past_guess_history"])
		if len(past_guess_history) == 0:
			# 首轮没有历史信息可利用，因此只给一个极小正奖励，避免它被无意义地视为“错误回合”。
			print("Uses previous feedback reward: 0.1 (No past guesses)")
			return 0.1

		# 三张“约束表”分别表达三类历史知识：
		# - correct_letter_to_position: 已确认某字母必须出现在某位置（绿色）
		# - valid_letter_to_position: 已确认某字母存在，但不能再放回这些位置（黄色）
		# - wrong_letter_to_position: 已经证伪的字母位置记录；当前实现里只要字母出现过灰色，就倾向于惩罚复用
		correct_letter_to_position = {}
		valid_letter_to_position = {}
		wrong_letter_to_position = {}
		for _, past_feedback in past_guess_history:
			past_feedback = past_feedback.split(" ")
			for i, fb in enumerate(past_feedback):
				if "✓" in fb:
					if fb[0] not in correct_letter_to_position:
						correct_letter_to_position[fb[0]] = set()
					correct_letter_to_position[fb[0]].add(i)
				elif "-" in fb:
					if fb[0] not in valid_letter_to_position:
						valid_letter_to_position[fb[0]] = set()
					valid_letter_to_position[fb[0]].add(i)
				else:
					if fb[0] not in wrong_letter_to_position:
						wrong_letter_to_position[fb[0]] = set()
					wrong_letter_to_position[fb[0]].add(i)

		# 对当前猜测逐字符打分。
		# 系数的相对强弱反映了策略优先级：
		# - 复用灰色字母最糟，因此给最大惩罚 -0.5
		# - 黄色字母放回已知错误位置属于“没有用到反馈”，给中等惩罚 -0.2
		# - 绿色字母保持在正确位置是最可靠的行为，每次 +0.2
		# - 黄色字母换到新位置是合理探索，每次 +0.1
		# - 完全不受约束的新字母给 +0.05，鼓励在安全前提下扩大信息覆盖面
		for idx, letter in enumerate(guess):
			if letter in correct_letter_to_position and idx in correct_letter_to_position[letter]:
				reward += 0.2
			elif letter in valid_letter_to_position and idx not in valid_letter_to_position[letter]:
				reward += 0.1
			elif letter in valid_letter_to_position and idx in valid_letter_to_position[letter]:
				reward -= 0.2
			elif letter in wrong_letter_to_position:
				reward -= 0.5
			else:
				reward += 0.05

	except Exception:
		# 反馈解析失败时宁可不给分，也不要把异常样本错误奖励成“高质量策略”。
		return 0.0

	return reward


__all__ = ["uses_previous_feedback"]

