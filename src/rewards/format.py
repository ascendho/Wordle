def output_format_check(prompt: str, completion: str, example: dict) -> int:
    """
    奖励函数：验证模型输出格式。
    
    判断模型的回答是否遵循 `<think>思考过程</think>
<guess>具体词语</guess>` 的严格格式，
    并且猜出的词语是否为合法的 5 字母单词。
    """
    import re
    import pandas as pd

    reward = 0
    try:
        # 添加合成的 <think>，因为这在 prompt 中已经预填充了一部分，
        # 为了让助手更容易通过正则验证，这里直接做拼接。
        completion = "<think>" + completion

        # 检查是否匹配我们期待的 XML 标签模式：
        # 先出现 <think> 区域，然后接上独立的 <guess> 区域。
        regex = (
            r"^<think>\s*([^<]*(?:<(?!/?think>)[^<]*)*)\s*<\/think>
"
            r"<guess>\s*([\s\S]*?)\s*<\/guess>$"
        )

        # 执行正则搜索匹配
        match = re.search(regex, completion, re.DOTALL)
        # 格式错误给予零分奖励
        if match is None or len(match.groups()) != 2:
            return 0

        # 取出预测词并去两端空格
        guess = match.groups()[1]
        guess = guess.strip()

        # 校验规则 1：单词必须正好是 5 个字符
        if len(guess) != 5:
            return 0.1

        # 校验规则 2：利用当前训练样本注入的词库判断是不是有效的真词汇
        word_list = pd.read_csv(str(example["word_list"]))
        if guess not in word_list["Word"].values:
            return 0.5  # 格式和长度对，但单词无效给予 0.5 分

        # 严格符合格式并给出合法预备词语，给予满分初始奖励
        reward = 1.0
    except Exception:
        pass

    return reward
