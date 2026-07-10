"""SFT prompt-completion 转换和 token 边界审计的单元测试。"""

from __future__ import annotations

from medical_grpo.data.schema import build_sft_messages
from medical_grpo.sft.data import audit_token_boundaries, build_hf_dataset, to_prompt_completion


class FakeChatTokenizer:
    """用字符 ID 模拟 chat template，便于精确测试监督边界。"""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        """生成可预测的 user 前缀和 assistant completion 边界。"""

        assert not tokenize
        text = ""
        for message in messages:
            if message["role"] == "user":
                text += f"<user>{message['content']}</user>"
            else:
                text += f"<assistant>{message['content']}</assistant>"
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, texts: list[str], **_: object) -> dict[str, list[list[int]]]:
        """把每个字符映射为唯一整数，避免引入真实 tokenizer 依赖。"""

        return {"input_ids": [[ord(character) for character in text] for text in texts]}

    def decode(self, token_ids: list[int], **_: object) -> str:
        """把字符 ID 还原为文本，供 supervised_preview 断言。"""

        return "".join(chr(token_id) for token_id in token_ids)


def _record() -> dict[str, object]:
    """构造满足 Huatuo canonical schema 的最小单轮样本。"""

    return {
        "id": "sft-1",
        "source": "huatuo_o1_sft_en",
        "question": "What is the diagnosis?",
        "reasoning": "The findings support pneumonia.",
        "response": "Pneumonia.",
        "messages": build_sft_messages(
            "What is the diagnosis?",
            "The findings support pneumonia.",
            "Pneumonia.",
        ),
        "split": "train",
        "meta": {},
    }


def test_prompt_completion_preserves_huatuo_headings() -> None:
    """转换后必须保留角色、标题和 TRL 所需四列。"""

    prepared = to_prompt_completion(_record())
    dataset = build_hf_dataset([_record()])

    assert prepared["prompt"][0]["role"] == "user"
    assert prepared["completion"][0]["role"] == "assistant"
    assert prepared["completion"][0]["content"].startswith("## Thinking")
    assert dataset.column_names == ["id", "source", "prompt", "completion"]


def test_token_audit_proves_prompt_prefix_and_completion_tokens() -> None:
    """监督边界应无错位、空 completion 或 max_length 截断。"""

    summary = audit_token_boundaries(
        [_record()],
        FakeChatTokenizer(),
        max_length=2048,
        inspect_samples=1,
    )

    assert summary.rows == 1
    assert summary.truncated_rows == 0
    assert summary.empty_completion_rows == 0
    assert summary.prefix_mismatch_rows == 0
    assert summary.min_supervised_tokens > 0
    assert summary.inspected_samples[0]["supervised_preview"].startswith("## Thinking")
