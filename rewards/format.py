def output_format_check(prompt: str, completion: str, example: dict) -> int:
    import re
    import pandas as pd

    reward = 0
    try:
        # Add synthetic <think> as it's already part of the prompt and prefilled 
        # for the assistant to more easily match the regex
        completion = "<think>" + completion

        # Check if the format matches expected pattern:
        # <think> content </think> followed by <answer> content </answer>
        regex = (
            r"^<think>\s*([^<]*(?:<(?!/?think>)[^<]*)*)\s*<\/think>\n"
            r"<guess>\s*([\s\S]*?)\s*<\/guess>$"
        )

        # Search for the regex in the completion
        match = re.search(regex, completion, re.DOTALL)
        if match is None or len(match.groups()) != 2:
            return 0

        guess = match.groups()[1]
        guess = guess.strip()

        # If the word is not 5 characters, return 0
        if len(guess) != 5:
            return 0.1

        # Check if the guess is a valid word compared to a predifined list of words
        word_list = pd.read_csv(str(example["word_list"]))
        if guess not in word_list["Word"].values:
            return 0.5

        reward = 1.0
    except Exception:
        pass

    return reward
