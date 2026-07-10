"""英文主线使用的模型和官方数据源 revision。

这里不接受浮动的 main 分支；源 revision 会同时写入 manifest，保证可追溯。
"""

from __future__ import annotations

from dataclasses import dataclass


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


@dataclass(frozen=True)
class DatasetSource:
    """一个可复现的 Hugging Face 数据源定位。"""

    key: str
    repository: str
    revision: str
    config: str | None
    split: str


# 训练数据只采用 HuatuoGPT-o1 官方 English SFT 和 verifiable problems；
# 三个 benchmark 只作为保护集和最终评测集合。
SOURCES: dict[str, DatasetSource] = {
    "huatuo_sft_en": DatasetSource(
        key="huatuo_sft_en",
        repository="FreedomIntelligence/medical-o1-reasoning-SFT",
        revision="fc2c9e8a37b38f38da6d449564a8c350b244aef4",
        config="en",
        split="train",
    ),
    "huatuo_verifiable": DatasetSource(
        key="huatuo_verifiable",
        repository="FreedomIntelligence/medical-o1-verifiable-problem",
        revision="46d5175eb74fdef3516d51d52e8c40db04bbdf35",
        config="default",
        split="train",
    ),
    "medqa_usmle": DatasetSource(
        key="medqa_usmle",
        repository="GBaker/MedQA-USMLE-4-options",
        revision="0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18",
        config=None,
        split="test",
    ),
    "medmcqa": DatasetSource(
        key="medmcqa",
        repository="openlifescienceai/medmcqa",
        revision="91c6572c454088bf71b679ad90aa8dffcd0d5868",
        config=None,
        split="validation",
    ),
    "pubmedqa_labeled": DatasetSource(
        key="pubmedqa_labeled",
        repository="qiaojin/PubMedQA",
        revision="9001f2853fb87cab8d220904e0de81ac6973b318",
        config="pqa_labeled",
        split="train",
    ),
}
