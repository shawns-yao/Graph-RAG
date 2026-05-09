"""Utility helpers copied from GraphRAG-Benchmark evaluation flow."""

from __future__ import annotations

import json
import re
from typing import Any

import json5
import json_repair


class JSONHandler:
    """Robust JSON parser with layered repair strategies."""

    def __init__(self, max_retries: int = 2, self_healing: bool = False) -> None:
        self.max_retries = max_retries
        self.self_healing = self_healing

    @staticmethod
    def safe_json_parse(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        try:
            return json5.loads(text)
        except Exception:
            pass
        try:
            repaired = json_repair.repair_json(text)
            return json.loads(repaired)
        except Exception:
            return {}

    @staticmethod
    def extract_json_block(text: str) -> str:
        match = re.search(r"\{[\s\S]*\}", text)
        return match.group(0) if match else text

    @staticmethod
    def extract_array_fallback(text: str) -> list[str]:
        match = re.search(r"\[([\s\S]*?)\]", text)
        if not match:
            return []
        items = re.split(r",\s*", match.group(1))
        return [item.strip(" \"'") for item in items if item.strip()]

    @staticmethod
    def validate_list(items: Any) -> list[Any]:
        if not isinstance(items, list):
            return []
        cleaned: list[Any] = []
        for item in items:
            if isinstance(item, str) and item.strip():
                cleaned.append(item.strip())
            elif isinstance(item, dict):
                cleaned.append(item)
        return cleaned

    async def parse_with_fallbacks(
        self,
        raw_text: str,
        key: str | None = None,
        llm: Any | None = None,
        callbacks: Any | None = None,
    ) -> list[str] | dict[str, Any]:
        del callbacks
        content = re.sub(r"```(?:json)?|```", "", raw_text).strip()

        data = self.safe_json_parse(content)
        if key and key in data:
            return self.validate_list(data[key])
        if not key and data:
            return data

        json_block = self.extract_json_block(content)
        data = self.safe_json_parse(json_block)
        if key and key in data:
            return self.validate_list(data[key])
        if not key and data:
            return data

        if key:
            fallback_array = self.extract_array_fallback(content)
            if fallback_array:
                return self.validate_list(fallback_array)

        if self.self_healing and llm is not None:
            healed = await self.heal_with_llm(raw_text, key, llm)
            if healed:
                return healed
        return [] if key else {}

    async def heal_with_llm(
        self,
        invalid_text: str,
        key: str | None,
        llm: Any,
    ) -> list[str] | dict[str, Any]:
        repair_prompt = f"""
Return ONLY valid JSON{f" with a key '{key}'" if key else ""}.
Invalid output was:
{invalid_text}
"""
        for _ in range(self.max_retries):
            try:
                response = await llm.ainvoke(repair_prompt)
                repaired_text = re.sub(r"```(?:json)?|```", "", response.content).strip()
                data = self.safe_json_parse(repaired_text)
                if key and key in data:
                    return self.validate_list(data[key])
                if not key and data:
                    return data
            except Exception:
                continue
        return [] if key else {}

