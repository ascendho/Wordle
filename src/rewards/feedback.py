"""奖励函数：评估模型是否合理利用了之前的 Wordle 反馈。"""

import ast
import re


def uses_previous_feedback(prompt: str, completion: str, example: dict) -> float:
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

		past_guess_history = ast.literal_eval(example["past_guess_history"])
		if len(past_guess_history) == 0:
			print("Uses previous feedback reward: 0.1 (No past guesses)")
			return 0.1

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
		return 0.0

	return reward


__all__ = ["uses_previous_feedback"]

