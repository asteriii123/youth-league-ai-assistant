"""
大语言模型调用模块。

负责：

1. 初始化DeepSeek客户端
2. 发送Prompt
3. 普通生成
4. 流式生成


被：

ai.py

meeting.py

调用。
"""
import json
from typing import Any

import httpx

from app.core.config import settings


class DeepSeekError(Exception):
    """Raised when the configured DeepSeek service cannot complete a request."""


class DeepSeekClient:
    def __init__(self) -> None:
        self.api_key = settings.deepseek_api_key
        self.base_url = settings.deepseek_base_url
        self.model = settings.deepseek_model

    async def complete(self, messages: list[dict[str, str]], *, json_output: bool = False) -> str:
        if not self.api_key:
            raise DeepSeekError("DEEPSEEK_API_KEY 尚未配置")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 2000,
            "stream": False,
        }
        if json_output:
            payload["response_format"] = {"type": "json_object"}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0)) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise DeepSeekError("DeepSeek 暂时无法完成请求，请稍后重试") from exc
        if not isinstance(content, str) or not content.strip():
            raise DeepSeekError("DeepSeek 返回了空内容，请稍后重试")
        return content.strip()

    async def stream(self, messages: list[dict[str, str]]):
        if not self.api_key:
            raise DeepSeekError("DEEPSEEK_API_KEY 尚未配置")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 2000,
            "stream": True,
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            content = json.loads(data)["choices"][0]["delta"].get("content", "")
                        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                            continue
                        if content:
                            yield content
        except httpx.HTTPError as exc:
            raise DeepSeekError("DeepSeek 暂时无法完成请求，请检查密钥和网络") from exc


def get_deepseek_client() -> DeepSeekClient:
    return DeepSeekClient()
