"""Tests for chat_messages module (ChatMessage, coerce_messages, transcript_to_messages)."""

from __future__ import annotations

from lifers.chat_messages import ChatMessage, coerce_messages, transcript_to_messages


class TestCoerceMessages:
    def test_empty_list(self) -> None:
        assert coerce_messages([]) == []

    def test_non_list_returns_empty(self) -> None:
        assert coerce_messages(None) == []
        assert coerce_messages("hello") == []
        assert coerce_messages({}) == []

    def test_valid_messages(self) -> None:
        msgs = coerce_messages([
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ])
        assert len(msgs) == 3
        assert msgs[0] == {"role": "system", "content": "You are a helper."}

    def test_skips_invalid_role(self) -> None:
        msgs = coerce_messages([
            {"role": "invalid", "content": "hello"},
            {"role": "user", "content": "valid"},
        ])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_skips_non_string_content(self) -> None:
        msgs = coerce_messages([
            {"role": "user", "content": 123},
            {"role": "user", "content": "valid text"},
        ])
        assert len(msgs) == 1
        assert msgs[0]["content"] == "valid text"

    def test_skips_non_dict_items(self) -> None:
        msgs = coerce_messages(["string", 42, {"role": "user", "content": "ok"}])
        assert len(msgs) == 1
        assert msgs[0]["content"] == "ok"

    def test_includes_name_field(self) -> None:
        msgs = coerce_messages([
            {"role": "user", "content": "hi", "name": "Alice"},
        ])
        assert msgs[0].get("name") == "Alice"

    def test_skips_empty_name(self) -> None:
        msgs = coerce_messages([
            {"role": "user", "content": "hi", "name": ""},
        ])
        assert "name" not in msgs[0]


class TestTranscriptToMessages:
    def test_basic_transcript(self) -> None:
        msgs = transcript_to_messages("sys", "usr", "ast")
        assert len(msgs) == 3
        assert msgs[0] == {"role": "system", "content": "sys"}
        assert msgs[1] == {"role": "user", "content": "usr"}
        assert msgs[2] == {"role": "assistant", "content": "ast"}

    def test_strips_whitespace(self) -> None:
        msgs = transcript_to_messages("  sys  ", "  usr  ", "  ast  ")
        assert msgs[0]["content"] == "sys"
        assert msgs[1]["content"] == "usr"
        assert msgs[2]["content"] == "ast"

    def test_types_are_correct(self) -> None:
        msgs = transcript_to_messages("a", "b", "c")
        assert isinstance(msgs[0], dict)
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
