"""
Ollama LLM 问答客户端 (Local LLM Client)
=========================================
通过 Ollama 本地服务调用大语言模型进行智能问答。
如果 Ollama 不可用, 系统自动回退到内置规则问答。

默认连接地址: http://127.0.0.1:11434
优先模型: deepseek-r1:7b, llama3:latest
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

# Ollama 默认服务地址
OLLAMA_URL = "http://127.0.0.1:11434"


@dataclass
class OllamaStatus:
    """Ollama 服务状态

    available: 服务是否可用
    model: 当前选择的模型名
    models: 服务端所有可用模型列表
    message: 状态描述
    """
    available: bool
    model: str | None
    models: list[str]
    message: str


class OllamaAgentClient:
    """Ollama 本地 LLM 客户端

    功能:
      1. 自动检测 Ollama 服务状态
      2. 从可用模型中选择最佳匹配
      3. 基于系统上下文进行中文问答

    使用方式:
      client = OllamaAgentClient()
      if client.status.available:
          answer = client.generate_answer("问题", context_dict)
    """

    def __init__(
        self,
        preferred_models: list[str] | None = None,
        base_url: str = OLLAMA_URL,
        timeout: int = 45,
    ):
        """
        Args:
            preferred_models: 偏好模型列表 (按优先级排序)
            base_url: Ollama 服务地址
            timeout: HTTP 请求超时秒数
        """
        self.preferred_models = preferred_models or ["deepseek-r1:7b", "llama3:latest"]
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # 初始化时自动检测服务状态
        self.status = self._detect_status()

    def _request_json(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """向 Ollama API 发送 HTTP 请求并解析 JSON 响应

        Args:
            path: API 路径 (如 "/api/tags", "/api/generate")
            payload: POST 请求体 (None 则为 GET)

        Returns:
            解析后的 JSON 字典
        """
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method="POST" if data else "GET",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _detect_status(self) -> OllamaStatus:
        """检测 Ollama 服务状态并选择可用模型

        检测逻辑:
          1. 调用 /api/tags 获取可用模型列表
          2. 从偏好模型列表中选择第一个可用的
          3. 如果没有偏好模型可用, 选择服务器上第一个模型
          4. 如果服务不可达, 标记为 unavailable
        """
        try:
            payload = self._request_json("/api/tags")
            models = [item["name"] for item in payload.get("models", [])]

            # 按优先级选择模型
            chosen = next(
                (model for model in self.preferred_models if model in models),
                models[0] if models else None,
            )

            if chosen:
                return OllamaStatus(
                    available=True,
                    model=chosen,
                    models=models,
                    message=f"Ollama available with model {chosen}",
                )
            return OllamaStatus(
                available=False,
                model=None,
                models=models,
                message="Ollama service is running but no model is available.",
            )
        except Exception as exc:
            return OllamaStatus(
                available=False,
                model=None,
                models=[],
                message=f"Ollama unavailable: {type(exc).__name__}",
            )

    def refresh_status(self) -> OllamaStatus:
        """重新检测 Ollama 服务状态 (用户手动触发重连或运行时恢复)"""
        self.status = self._detect_status()
        return self.status

    def generate_answer(self, question: str, context: dict[str, Any]) -> dict[str, Any] | None:
        """使用 Ollama LLM 生成问答回复

        通过精心设计的 System Prompt 和 User Prompt:
          - 约束模型使用中文回答
          - 要求基于提供的系统上下文, 不编造数据
          - 优先解释预测、风险、调度策略等内容

        Args:
            question: 用户问题
            context: 系统上下文数据 (summary_cards, station_overview 等)

        Returns:
            {"answer": "...", "backend": "ollama", "model": "deepseek-r1:7b"}
            如果服务不可用或生成失败, 返回 None
        """
        if not self.status.available or not self.status.model:
            # 自动重试: 初次检测可能因为 Ollama 未就绪而失败, 实际调用时再试一次
            self.status = self._detect_status()
            if not self.status.available or not self.status.model:
                return None

        # System Prompt: 定义智能体的角色和能力边界
        system_prompt = (
            "你是一个海上风电预测与风险调控智能体助手。"
            "请基于提供的系统结果, 用简洁、专业、适合课程答辩的中文回答。"
            "不要编造数据; 如果上下文中没有, 就明确说当前系统没有提供。"
            "优先解释预测结果、风险原因、强化学习动作逻辑、站点差异。"
        )

        # User Prompt: 问题 + 上下文数据
        user_prompt = (
            f"用户问题：{question}\n\n"
            f"系统上下文：\n{json.dumps(context, ensure_ascii=False)}\n\n"
            "请输出一段自然中文, 不要使用 markdown 标题。"
        )

        try:
            payload = self._request_json(
                "/api/generate",
                {
                    "model": self.status.model,
                    "system": system_prompt,
                    "prompt": user_prompt,
                    "stream": False,  # 非流式, 等待完整回复
                    "options": {
                        "temperature": 0.3,  # 低温度让回答更稳定、严谨
                    },
                },
            )
            answer = str(payload.get("response", "")).strip()
            if not answer:
                return None
            return {
                "answer": answer,
                "backend": "ollama",
                "model": self.status.model,
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            # 任何网络/解析错误 → 返回 None, 由调用方回退到规则问答
            return None
