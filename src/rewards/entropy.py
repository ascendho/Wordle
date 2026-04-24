"""奖励函数：计算最新猜词的归一化信息增益。"""

import ast
import math
import re

import pandas as pd


def _validate_guess(secret: str, guess: str, raw_feedback: bool = False):
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
	filtered = []
	for word in all_candidate_words:
		valid = True
		for past_guess, past_feedback in past_guesses:
			candidate_feedback = _validate_guess(word, past_guess)
			if candidate_feedback != past_feedback:
				valid = False
				break
		if valid:
			filtered.append(word)
	return filtered


def _compute_normalized_information_gain(all_candidate_words, past_guesses, guess):
	candidates = _filter_candidates(all_candidate_words, past_guesses)
	total_candidates = len(candidates)
	if total_candidates == 0:
		return 0.0, 0.0

	current_entropy = math.log2(total_candidates)
	feedback_groups = {}
	for word in candidates:
		feedback = _validate_guess(word, guess, raw_feedback=True)
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
		group_entropy = math.log2(group_size) if group_size > 0 else 0.0
		expected_entropy += p * group_entropy
		info_gain = current_entropy - group_entropy
		max_info_gain = max(max_info_gain, info_gain)

	expected_gain = current_entropy - expected_entropy
	normalized_expected_gain = expected_gain / current_entropy if current_entropy > 0 else 0.0
	normalized_max_gain = max_info_gain / current_entropy if current_entropy > 0 else 0.0
	return normalized_expected_gain, normalized_max_gain


def guess_value(prompt: str, completion: str, example: dict) -> float:
	reward = 0.0
	try:
		# Add synthetic <think> as it's already part of the prompt and prefilled
		# for the assistant to more easily match the regex.
		completion = "<think>" + completion

		regex = r"<guess>\s*([\s\S]*?)\s*<\/guess>$"
		match = re.search(regex, completion, re.DOTALL)
		if match is None or len(match.groups()) != 1:
			return 0.0

		guess = match.groups()[0].strip()
		if len(guess) != 5:
			return 0.0

		word_list = pd.read_csv(str(example["word_list"]))
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
		return 0.0

	return reward


__all__ = ["guess_value"]

