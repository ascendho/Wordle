# Reward function that checks if the guess uses the previous feedback for its next guess
def uses_previous_feedback(prompt: str, completion: str, example: dict) -> float:
    """
    奖励函数：评估模型是否在下次猜测中合理利用了之前的 Wordle 线索反馈。
    它会根据过去的反馈，对猜测中的对、错、部分正确字母进行奖惩。
    """
    import re
    import ast

    reward = 0.0
    try:
        # 将原始生成的文本补充合成的 <think> 标签以便解析
        completion = "<think>" + completion

        # 匹配 <guess>...</guess> 提取最新的猜测单词
        regex = r"<guess>\s*([\s\S]*?)\s*<\/guess>$"
        match = re.search(regex, completion, re.DOTALL)
        if match is None or len(match.groups()) != 1:
            return 0.0

        guess = match.groups()[0].strip()
        # 若不是有效的5字结构则不获得这部分环境反馈分
        if len(guess) != 5:
            return 0.0

        # 安全地将字符串转化为列表提取此前的游戏记录 
        past_guess_history = ast.literal_eval(example["past_guess_history"])
        
        # 若本轮未曾猜过（首轮猜测），则基于不触发惩罚的原则给予底分
        if len(past_guess_history) == 0:
            print("Uses previous feedback reward: 0.1 (No past guesses)")
            return 0.1

        # 定义分类字典汇集已知的对错位置信息
        correct_letter_to_position = {} # 绿色：字母存在且位置对
        valid_letter_to_position = {}   # 黄色：字母存在但位置错
        wrong_letter_to_position = {}   # 灰色：排除了该字母

        # 遍历过去的反馈历史重建游戏盘面线索状态
        for _, past_feedback in past_guess_history:
            past_feedback = past_feedback.split(" ")
            for i, fb in enumerate(past_feedback):
                if '✓' in fb:
                    if fb[0] not in correct_letter_to_position:
                        correct_letter_to_position[fb[0]] = set()
                    correct_letter_to_position[fb[0]].add(i)
                elif '-' in fb:
                    if fb[0] not in valid_letter_to_position:
                        valid_letter_to_position[fb[0]] = set()
                    valid_letter_to_position[fb[0]].add(i)
                else:
                    if fb[0] not in wrong_letter_to_position:
                        wrong_letter_to_position[fb[0]] = set()
                    wrong_letter_to_position[fb[0]].add(i)

        # 遍历当前新的 guess，给字母对应的状况发奖励或进行惩罚扣分
        for idx, letter in enumerate(guess):
            # 奖励：重用了已知处于该位置的对应正确字母
            if (letter in correct_letter_to_position and idx in correct_letter_to_position[letter]):
                reward += 0.2
            # 奖励：使用了已知存在的字母，并且是在新的位置探索
            elif (letter in valid_letter_to_position and idx not in valid_letter_to_position[letter]):
                reward += 0.1
            # 惩罚：复用了一个已知在错误位置存在的字母，没有形成新的排查
            elif (letter in valid_letter_to_position and idx in valid_letter_to_position[letter]):
                reward -= 0.2
            # 严重惩罚：复用被明确排除了的灰色字母
            elif letter in wrong_letter_to_position:
                reward -= 0.5
            else:
                # 给全新字母微小正奖励鼓动探索机制
                reward += 0.05

    except Exception:
        # 当发生解析故障安全回退
        return 0.0

    return float(reward)
