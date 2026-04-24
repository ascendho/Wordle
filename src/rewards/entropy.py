"""奖励函数：计算最新猜词在当前候选词空间上的归一化信息增益。"""

import ast
import math
import re

import pandas as pd


def _validate_guess(secret: str, guess: str, raw_feedback: bool = False):
	"""
	按照 Wordle 规则生成 `guess` 相对 `secret` 的反馈。

	这里采用两遍扫描：
	1. 第一遍先处理绿色命中，优先锁定“字母和位置都正确”的格子。
	2. 第二遍再处理黄色/灰色，借助被置空的 `secret_list` 正确处理重复字母。

	raw_feedback=True 时返回逐位置列表，便于后续进一步编码；否则返回训练数据同款字符串。
	"""
	feedback = []
	secret_list = list(secret)

	for i, (g_char, s_char) in enumerate(zip(guess, secret)):
		if g_char == s_char:
			feedback.append(f"{g_char}(✓) ")
			secret_list[i] = None
		else:
			feedback.append(None)

	for i, g_char in enumerate(guess):
		if feedback[i] is None:
			if g_char in secret_list:
				feedback[i] = f"{g_char}(-) "
				secret_list[secret_list.index(g_char)] = None
			else:
				feedback[i] = f"{g_char}(x) "

	if raw_feedback:
		return feedback
	return "".join(feedback).strip()


def _filter_candidates(all_candidate_words, past_guesses):
	"""用历史猜测及其反馈回放约束，筛出仍然可能成为 secret 的候选词。"""
	filtered = []
	for word in all_candidate_words:
		valid = True
		for past_guess, past_feedback in past_guesses:
			# 如果把当前 candidate 当作真实 secret 时，生成不出同样的历史反馈，
			# 说明它已经被过去回合排除了。
			candidate_feedback = _validate_guess(word, past_guess)
			if candidate_feedback != past_feedback:
				valid = False
				break
		if valid:
			filtered.append(word)
	return filtered


def _compute_normalized_information_gain(all_candidate_words, past_guesses, guess):
	"""
	估计 `guess` 在当前候选空间中的期望信息增益，并做归一化。

	核心思路：
	1. 先利用历史反馈缩小候选集。
	2. 假设候选集中的每个词都有可能是真实答案，计算 `guess` 会诱导出的反馈模式。
	3. 不同反馈模式会把候选集切成若干组；组越碎，说明这次猜测越能减少不确定性。
	4. 用熵差 `current_entropy - expected_entropy` 衡量平均信息收益，再除以当前熵得到 0~1 附近的归一化分数。

	额外返回的 `normalized_max_gain` 是理论上某个单一反馈分支能带来的最大归一化收益，
	当前训练没有直接使用，但保留下来便于后续分析或扩展奖励设计。
	"""
	candidates = _filter_candidates(all_candidate_words, past_guesses)
	total_candidates = len(candidates)
	if total_candidates == 0:
		return 0.0, 0.0

	# 当前熵代表“在不知道下一次反馈前，我们对答案还剩多少不确定性”。
	current_entropy = math.log2(total_candidates)
	feedback_groups = {}
	for word in candidates:
		feedback = _validate_guess(word, guess, raw_feedback=True)
		# 为了只关心反馈结构而非具体字母，这里把绿色/黄色/灰色压缩编码成 1/0/x。
		# 同一模式下的候选词会落入同一个桶，它们对应“模型观察到相同反馈后仍无法区分”的剩余空间。
		feedback_pattern = "".join(
			"1" if "✓" in fb else ("0" if "-" in fb else "x")
			for fb in feedback
		)
		feedback_groups.setdefault(feedback_pattern, []).append(word)

	expected_entropy = 0.0
	max_info_gain = 0.0
	for group in feedback_groups.values():
		group_size = len(group)
		p = group_size / total_candidates
		# 某个反馈分支发生后，剩余的不确定性就是该分支组内候选词数的对数熵。
		group_entropy = math.log2(group_size) if group_size > 0 else 0.0
		expected_entropy += p * group_entropy
		info_gain = current_entropy - group_entropy
		max_info_gain = max(max_info_gain, info_gain)

	# 期望收益反映“平均而言，这个猜词能把搜索空间压缩多少”。
	expected_gain = current_entropy - expected_entropy
	normalized_expected_gain = expected_gain / current_entropy if current_entropy > 0 else 0.0
	normalized_max_gain = max_info_gain / current_entropy if current_entropy > 0 else 0.0
	return normalized_expected_gain, normalized_max_gain


def guess_value(prompt: str, completion: str, example: dict) -> float:
	"""
	返回当前猜词的归一化期望信息增益，鼓励模型优先选择“最能缩小候选空间”的词。

	与 format/feedback 奖励不同，这里关注的是搜索效率：
	即使一个词暂时没猜中答案，只要它能显著区分候选集，就应该得到更高奖励。
	"""
	reward = 0.0
	try:
		# 与其它奖励函数保持同样的 completion 预处理方式。
		completion = "<think>" + completion

		regex = r"<guess>\s*([\s\S]*?)\s*<\/guess>$"
		match = re.search(regex, completion, re.DOTALL)
		if match is None or len(match.groups()) != 1:
			return 0.0

		guess = match.groups()[0].strip()
		if len(guess) != 5:
			return 0.0

		word_list = pd.read_csv(str(example["word_list"]))
		# 非法词不值得做信息论评估，因为真实环境根本不会接受这样的猜测。
		if guess not in word_list["Word"].values:
			return 0.0

		past_guess_history = ast.literal_eval(example["past_guess_history"])
		normalized_expected_gain, _ = _compute_normalized_information_gain(
			word_list["Word"].values,
			past_guess_history,
			guess,
		)
		reward = normalized_expected_gain
	except Exception:
		# 任意解析/读词表失败都按 0 分处理，避免噪声样本破坏奖励稳定性。
		return 0.0

	return reward


__all__ = ["guess_value"]

