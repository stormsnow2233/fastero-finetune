# Qwen2.5-3B-Instruct 中文聊天 SFT — 数据与训练配置

基础模型 Qwen2.5-3B-Instruct，方法 QLoRA，目标：自然中文多轮聊天。
Fastero 没有 GPU/训练功能，所以数据和配置在本机准备好，训练在 Colab 免费 T4 上跑。

## 数据源

**NaturalConv**（腾讯 AI Lab，AAAI 2021，论文 arXiv:2103.02548）
- 19,919 条人工采集的中文多轮主题对话，约 400K 轮，平均 20.1 轮/对话，6 个领域。
- 官方发布在腾讯 AI Lab；本机用的镜像：HuggingFace `xywang1/NaturalConv`（861 下载，22 likes）。
- GitHub 上没有官方数据仓库（`thu-coai/NaturalConv` 不存在；`naturalconv/NaturalConvDataSet` 只有下载文档代码，25 星）。
- **许可证：腾讯 AI Lab NaturalConv 数据条款——仅限个人非商业使用**（科研、开发、测试；不可商用、不可再分发）。个人学习项目可用，商用需联系 ailab@tencent.com。
- 备选对比：LCCC（thu-coai/CDial-GPT，1977 星，MIT，680 万条）是微博短文本对话，轮数少、口语化但偏碎片，不如 NaturalConv 贴合“自然多轮聊天”，故未采用。

## 清洗与抽样（clean_data.py，seed 42）

- 原始 19,919 条 → 去重 21 条 → 质量过滤剔除 21 条 → 可用池 19,334 条（train split）/ 272 条（dev split）。
- 过滤规则：单条消息 >200 字、同一字符连续 8 次以上、极端脏话词、含 [图片]/[表情]/链接、超半数消息 ≤2 字、中文占比 <50%；SFT 样本截成偶数轮（以 assistant 回复结尾）。
- 抽样：**训练集 3,500 条**（来自官方 train split）+ **验证集 200 条**（来自官方 dev split，无泄漏）。

## 数据统计

| 指标 | 数值 |
|---|---|
| 原始条数 | 19,919 |
| 清洗后可用 | 19,606（train 池 19,334 + dev 池 272） |
| 训练集 / 验证集 | 3,500 / 200 |
| 平均轮数 | 20.05（20–24 轮） |
| 平均消息长度 | 20.6 字 |
| 平均对话长度 | 414 字 ≈ 259 token |
| 训练集总量 | ≈ 906K token / epoch，3 epoch ≈ 2.7M token |

## 文件

- `raw/dialog_release.json` — 原始数据（51.7 MB）+ train/dev/test split、LICENSE、README
- `data/train.json` — 训练集，ShareGPT 格式（9.0 MB）
- `data/validation.json` — 验证集，ShareGPT 格式（511 KB）
- `data/train_alpaca.json` — 训练集 alpaca 格式备用（35,096 条，按 assistant 轮展开，历史截最近 6 轮，20.6 MB）
- `dataset_info.json` — LlamaFactory 数据集注册文件
- `llamafactory_qlora.yaml` — 训练配置（每个关键参数都有中文注释说明）
- `train_colab.ipynb` — Colab 可直接跑的 notebook（装 LlamaFactory → 挂 Drive → 训 → adapter 存回 Drive → 快速验证）
- `clean_data.py` / `build_notebook.py` — 数据清洗 / notebook 生成脚本（可复现）
- `data_stats.json` — 统计数字

## 训练配置要点（llamafactory_qlora.yaml）

- 4-bit NF4 量化底座 + LoRA（rank 16，alpha 32，训全部线性层），lr 1e-4，3 epoch。
- 有效 batch 16（per-device 4 × 累积 4），cutoff 1024，bf16，梯度检查点，8-bit 分页 AdamW，余弦衰减 + 3% 热身。
- 约 656 个优化步，每 200 步存 checkpoint 并在验证集上评估。

## T4 估算

- 显存：4-bit 底座 ~2GB + LoRA/优化器/梯度 <1GB + 激活（梯度检查点，batch 4×1024）~3–4GB → 峰值约 6–7GB，**16GB T4 够用，还有一半余量**。
- 时间：2.7M token，T4 上 QLoRA 约 1500–2500 token/s → **约 30–60 分钟**（含评估与 checkpoint 开销）。
- adapter 约 50MB，存 Drive 即可。

## 怎么跑

1. 把 `data/train.json` 和 `data/validation.json` 上传到 Google Drive 的 `fastero-finetune/` 文件夹。
2. 在 Colab 打开 `train_colab.ipynb`（运行时选 T4 GPU），从上往下依次执行。
3. 训完 adapter 在 Drive 的 `fastero-finetune/adapter/`，notebook 最后一格会加载它聊两句验证效果。
