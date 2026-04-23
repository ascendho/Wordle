# Wordle 模型微调流水线 (Wordle SFT & GRPO Pipeline)

本项目是一个模块化的 Python 应用，旨在基于 **Qwen 2.5 7B Instruct** 和 **Predibase** 平台，对语言模型进行 Wordle 游戏的监督微调（SFT）与生成式奖励策略优化（GRPO）。

通过这套核心架构，模型不仅能学习基础的猜测格式（SFT），更可以通过强化学习机制（GRPO）在游戏尝试中获取奖励与惩罚，主动进化出高效解题、归纳词语的高级信息增益策略。

## 项目结构
项目采用了混合型的设计结构：
- **顶层目录**：存放流水线的核心触发点（如 `main.py` 主入口，`train/` 训练逻辑封装，与 `eval/` 评估脚本）。
- **`src/` 内部区**：包含各类业务复用模块代码，如数据集加载器、自定义 GRPO 强化奖励函数模型等。

```text
.
├── main.py                 # 项目统一 CLI 主入口
├── train/                  # 训练相关任务下发
│   ├── sft.py              # SFT（监督微调）
│   ├── grpo.py             # 纯 GRPO 训练
│   └── sftgrpo.py          # 混合训练（在 SFT 检查点上继续 GRPO）
├── eval/
│   └── evaluate.py         # 评估脚本，预测并在 Wordle 环境中算分
└── src/
    ├── data/
    │   └── loader.py       # 加载 HuggingFace 数据集到 Predibase
    ├── rewards/            # GRPO 强化的奖励函数群
    │   ├── format.py       # XML 响应格式奖励校验
    │   ├── feedback.py     # Wordle 游戏规则与反馈推断校验
    │   └── entropy.py      # （信息增益）探索与减少解空间不确定性的高级激励
    └── utils/
        └── config.py       # 配置拉取（如平台 API 密匙）
```

## 环境配置
1. 复制 `.env.example` 并重命名为 `.env`，填入你的 `PREDIBASE_API_KEY`：
   ```bash
   cp .env.example .env
   # 并编辑 .env 填写具体密钥
   ```
2. 安装环境依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 执行流程

整个项目业务的执行皆依赖于统一终端接入点—— `main.py`。请按以下顺序执行流水线。

### 1. 数据准备
首先需要将存储于 HuggingFace 上的所需数据集迁移并缓存到 Predibase 云平台上：
```bash
python main.py --run data
```

### 2. 触发分布式训练流
当数据就绪后，可以根据你的策略来单独或联合下发训练作业。

- **阶段一：启动监督任务 (SFT)**
  使模型熟悉 `<think>` 推理层与 `<guess>` 打包层的作用。
  ```bash
  python main.py --run train --type sft
  ```

- **阶段二：启动强化探索 (GRPO)**
  仅依赖内置奖励机制尝试找到最佳打法（适合强基座模型进行 zero-shot exploratory）。
  ```bash
  python main.py --run train --type grpo
  ```

- **组合策略 (SFT + GRPO)**
  （推荐策略）从上述阶段一半成品的 SFT 版本继续强化探索：
  ```bash
  python main.py --run train --type sftgrpo --adapter wordle/1
  ```
  *(注：`--adapter` 表示承接的存档标识名)*

### 3. 评估与基准测试
使用测试数据集及对应已完成训练的 Adapter 模型，推演实际玩 Wordle 时的胜率和平均使用的轮数，校验模型智力并打印为清晰报表。
```bash
python main.py --run eval --adapter wordle/1
```

## 实验结果

通过执行 `eval` 过程可以得到如下展示的数据统计指标（成功通关数与胜组平均轮数）。对比多个流行大模型与我们自行微调的模型基准表现如下：

| 模型方案 | 成功通关数 (10 局制) | 获胜组内平均猜词数 |
| :--- | :---: | :---: |
| GPT-4o-mini | 1 | 4 |
| Claude 3.5 Sonnet | 8 | 4 |
| Claude 3.7 Sonnet Thinking (8K Budget) | 10 | 3.9 |
| Qwen 2.5 7B Instruct (Base 基础版) | 0 | 6 |
| Qwen 2.5 7B Instruct (纯 GRPO) | 3 | 4 |
| **Qwen 2.5 7B Instruct (SFT + GRPO)** | **7** | **4** |

**结果分析与总结**：
如上表所示，拥有巨大参数量的闭源模型如 Claude 3.7 Sonnet 能在带有深度思考预算的情况下达到 10 次全通关（平均 3.9 步解决）。
然而，仅有 7B 级别参数量的 **Qwen 2.5 Instruct 基础版** 在没有任何专业提示词辅助情况下的“原生战力”为 0 次通关。
令人惊喜的是，当为其应用本项目的 **SFT + GRPO** 联合强化微调流程后，其通关数激增至 **7 次**（且能在平均 **4 步**以内破题），直接超越了原生 GPT-4o-mini 的表现，并大幅逼近最顶尖大模型的最优成绩。这展现出了小模型在垂直任务中利用强化学习极大释放潜能的特性。
