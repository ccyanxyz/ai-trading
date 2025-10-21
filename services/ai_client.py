"""LLM client abstractions for AI strategies."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from openai import OpenAI

try:  # pragma: no cover - optional import depending on openai version
    from openai import BadRequestError
except ImportError:  # pragma: no cover
    BadRequestError = Exception  # type: ignore[assignment]

def classify_openai_model(model: str) -> Tuple[bool, str, bool]:
    lowered = model.lower()
    if lowered.startswith("gpt-5"):
        return True, "max_output_tokens", False
    if lowered.startswith(("o4", "o3", "o1")):
        return True, "max_completion_tokens", False
    return False, "max_tokens", True


def to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    allowed_passthrough = {
        "input_text",
        "input_image",
        "input_file",
        "audio",
        "video",
        "computer_screenshot",
        "summary_text",
    }
    for message in messages:
        role = message.get("role", "user")
        raw_content = message.get("content", "")
        content_items: List[Dict[str, Any]] = []

        if isinstance(raw_content, str):
            content_items.append({"type": "input_text", "text": raw_content})
        elif isinstance(raw_content, (list, tuple)):
            for piece in raw_content:
                if isinstance(piece, dict) and piece.get("type") in allowed_passthrough:
                    content_items.append(piece)
                elif isinstance(piece, str):
                    content_items.append({"type": "input_text", "text": piece})
                elif isinstance(piece, dict) and "text" in piece:
                    content_items.append(
                        {"type": "input_text", "text": str(piece.get("text", ""))}
                    )
                else:
                    content_items.append({"type": "input_text", "text": str(piece)})
        else:
            content_items.append({"type": "input_text", "text": str(raw_content)})

        converted.append({"role": role, "content": content_items})

    return converted


def _collect_text_fragments(value: Any, bucket: List[str], seen: set) -> None:
    if value is None:
        return

    if isinstance(value, str):
        bucket.append(value)
        return

    marker = id(value)
    if marker in seen:
        return
    seen.add(marker)

    if isinstance(value, dict):
        for key in ("text", "content", "output_text", "value"):
            if key in value:
                _collect_text_fragments(value[key], bucket, seen)
        for child in value.values():
            if isinstance(child, (list, tuple, set, dict)):
                _collect_text_fragments(child, bucket, seen)
        return

    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_text_fragments(item, bucket, seen)
        return

    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            _collect_text_fragments(dumped, bucket, seen)
            return
        except Exception:
            pass

    if hasattr(value, "dict"):
        try:
            dumped = value.dict()
            _collect_text_fragments(dumped, bucket, seen)
            return
        except Exception:
            pass

    bucket.append(str(value))


def extract_message_text(message: Any) -> str:
    if message is None:
        return ""

    content = getattr(message, "content", message)

    if (not content or content == "") and hasattr(message, "model_dump"):
        dumped = message.model_dump()
        if isinstance(dumped, dict) and dumped.get("content"):
            content = dumped["content"]

    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict) and extra.get("content"):
        content = extra.get("content")

    collected: List[str] = []
    _collect_text_fragments(content, collected, set())
    return "\n".join(part for part in (frag.strip() for frag in collected) if part)


def extract_responses_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    output = getattr(resp, "output", None)
    if output:
        fragments: List[str] = []
        _collect_text_fragments(output, fragments, set())
        joined = "\n".join(part.strip() for part in fragments if part and part.strip())
        if joined:
            return joined
    return ""


class AIClient(Protocol):
    async def invoke(
        self,
        messages: List[Dict[str, Any]],
        *,
        parser: Callable[[Optional[str]], Any],
    ) -> Tuple[Optional[str], Any]:
        """Execute an AI call and return (response_id, parsed_payload)."""


class OpenAIClient:
    """Thin wrapper around OpenAI APIs supporting Responses and Chat."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        logger,
    ) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._logger = logger
        (
            self._use_responses,
            self._token_param,
            self._include_temperature,
        ) = classify_openai_model(model)

    async def invoke(
        self,
        messages: List[Dict[str, Any]],
        *,
        parser: Callable[[Optional[str]], Any],
    ) -> Tuple[Optional[str], Any]:
        if messages:
            system_message = messages[0]
            if isinstance(system_message, dict) and system_message.get("role") == "system":
                system_message["cache_control"] = {"type": "ephemeral"}
        if self._use_responses:
            kwargs: Dict[str, Any] = {
                "model": self._model,
                "input": to_responses_input(messages),
            }
            token_param = (self._token_param or "").strip()
            if token_param:
                if self._max_tokens:
                    kwargs[token_param] = self._max_tokens
                elif token_param == "max_output_tokens":
                    kwargs[token_param] = 10000
            if self._include_temperature and self._temperature is not None:
                kwargs["temperature"] = self._temperature
            try:
                resp = await asyncio.to_thread(self._client.responses.create, **kwargs)
            except BadRequestError as exc:  # pragma: no cover - network/auth issues
                self._logger.error("OpenAI Responses 调用失败: %s", exc)
                raise
            text = extract_responses_text(resp)
            return resp.id if resp else None, parser(text)

        kwargs = {
            "model": self._model,
            "messages": messages,
        }
        if self._include_temperature and self._temperature is not None:
            kwargs["temperature"] = self._temperature
        token_param = (self._token_param or "").strip()
        if token_param and self._max_tokens:
            kwargs[token_param] = self._max_tokens
        try:
            response = await asyncio.to_thread(self._client.chat.completions.create, **kwargs)
        except BadRequestError as exc:  # pragma: no cover - network/auth issues
            self._logger.error("OpenAI ChatCompletions 调用失败: %s", exc)
            raise
        message = response.choices[0].message
        text = extract_message_text(message)
        return response.id if response else None, parser(text)


__all__ = [
    "AIClient",
    "OpenAIClient",
    "classify_openai_model",
    "to_responses_input",
    "extract_message_text",
    "extract_responses_text",
]
