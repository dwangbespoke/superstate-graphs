"""Full-context, restartable JSON inference for graph induction.

Requests are never shortened. Successful responses are content-addressed by the
entire request and model revision. Invalid/truncated responses cannot enter the
cache. Cache files retain usage and model provenance for research accounting.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI


class InvalidModelJSON(ValueError):
    """The model did not return a complete JSON object."""


def parse_json_object(content: str) -> dict[str, Any]:
    """Accept plain JSON or one explicit Markdown JSON fence, never partial JSON."""
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        _, separator, text = text.partition("\n")
        if not separator:
            raise InvalidModelJSON("Incomplete Markdown JSON fence")
        text = text[:-3].strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidModelJSON(f"Invalid JSON at character {exc.pos}") from exc
    if not isinstance(result, dict):
        raise InvalidModelJSON("Expected a JSON object")
    return result


def qwen_generation_policy(
    *, thinking: bool, max_tokens: int, temperature: float | None = None,
    thinking_token_budget: int | None = None, top_p: float | None = None,
    top_k: int | None = None, presence_penalty: float | None = None,
) -> dict[str, Any]:
    """Resolve the recorded Qwen decode policy without changing input context.

    Thinking uses Qwen's precise-task sampling preset and a separate reasoning
    allowance, leaving output room for the required JSON. Explicit values win.
    Non-thinking requests retain greedy decoding unless explicitly overridden.
    """
    result: dict[str, Any] = {
        "temperature": (0.6 if thinking else 0.0) if temperature is None else temperature,
    }
    options = {
        "thinking_token_budget": thinking_token_budget,
        "top_p": top_p, "top_k": top_k, "presence_penalty": presence_penalty,
    }
    if thinking:
        defaults = {"thinking_token_budget": min(4096, max_tokens // 2),
                    "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}
        options = {name: defaults[name] if value is None else value
                   for name, value in options.items()}
    result.update({name: value for name, value in options.items() if value is not None})
    return result


class GraphLLM:
    """Async OpenAI-compatible client with durable caching and bounded retries.

    Example::

        async with GraphLLM() as llm:
            result = await llm.complete_json(
                [{"role": "user", "content": "Return {\"ok\": true}."}],
                max_tokens=128,
            )

    One instance should be used within one event loop. For synchronous GEPA
    callbacks, create and own a persistent loop or call ``complete_json_sync``.
    """

    def __init__(
        self,
        runtime_path: str | Path = "results/runtime/graph.json",
        cache_dir: str | Path = "results/cache/graph_llm",
        *,
        concurrency: int = 32,
        timeout: float = 900,
        attempts: int = 6,
        transient_attempts: int = 3,
    ) -> None:
        if concurrency < 1 or attempts < 1 or transient_attempts < 1 or timeout <= 0:
            raise ValueError("Concurrency, attempts, and timeout must be positive")
        self.runtime = json.loads(Path(runtime_path).read_text())
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = AsyncOpenAI(
            api_key=self.runtime["api_key"],
            base_url=self.runtime["api_base"],
            timeout=httpx.Timeout(timeout, connect=20, write=120, pool=60),
            max_retries=0,
        )
        self._semaphore = asyncio.Semaphore(concurrency)
        self._inflight: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self.attempts = attempts
        self.transient_attempts = transient_attempts
        self._replicas: list[GraphLLM] = [self]
        self.stats = {"requests": 0, "cache_hits": 0, "prompt_tokens": 0,
                      "completion_tokens": 0, "retries": 0}

    async def __aenter__(self) -> GraphLLM:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.close()

    def shard(self, route_key: str) -> GraphLLM:
        """Match the pool interface when only one model server is needed."""
        return self

    async def token_count(
        self, messages: list[dict[str, Any]], *, thinking: bool = False
    ) -> int:
        """Ask the actual serving tokenizer to size a composite full-context request."""
        url = self.runtime["api_base"].removesuffix("/v1").rstrip("/") + "/tokenize"
        payload = {
            "model": self.runtime["model"],
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                url, json=payload,
                headers={"Authorization": f"Bearer {self.runtime['api_key']}"},
            )
            response.raise_for_status()
            return int(response.json()["count"])

    async def complete_json(
        self,
        messages: list[dict[str, Any]],
        *,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        temperature: float | None = None,
        thinking: bool = False,
        thinking_token_budget: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        seed: int = 17,
        cache_namespace: str = "graph-v1",
    ) -> dict[str, Any]:
        """Return a complete JSON object; raise instead of truncating history.

        ``schema`` is a JSON Schema, not an OpenAI response_format wrapper.
        Increase ``max_tokens`` for graph edits; unfinished outputs are retried
        with twice that budget, capped at 32768 output tokens.
        """
        if not messages or max_tokens <= 0:
            raise ValueError("Nonempty messages and positive max_tokens are required")
        policy = qwen_generation_policy(
            thinking=thinking, max_tokens=max_tokens, temperature=temperature,
            thinking_token_budget=thinking_token_budget, top_p=top_p, top_k=top_k,
            presence_penalty=presence_penalty,
        )
        thinking_token_budget = policy.get("thinking_token_budget")
        if thinking_token_budget is not None and (
            not thinking or thinking_token_budget < 0 or thinking_token_budget >= max_tokens
        ):
            raise ValueError("A thinking budget requires thinking=True and room for final output")
        response_format: dict[str, Any] = {"type": "json_object"}
        if schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "graph_response", "strict": True, "schema": schema},
            }
        request = {
            "model": self.runtime["model"],
            "messages": messages,
            "temperature": policy["temperature"],
            "max_tokens": max_tokens,
            "seed": seed,
            "response_format": response_format,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}},
        }
        for name in ("top_p", "presence_penalty"):
            if name in policy:
                request[name] = policy[name]
        for name in ("top_k", "thinking_token_budget"):
            if name in policy:
                request["extra_body"][name] = policy[name]
        fingerprint = {
            "namespace": cache_namespace,
            "model_revision": self.runtime.get("revision"),
            "request": request,
        }
        encoded = json.dumps(fingerprint, sort_keys=True, ensure_ascii=False).encode()
        key = hashlib.sha256(encoded).hexdigest()
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            self.stats["cache_hits"] += 1
            return json.loads(path.read_text())["result"]
        existing = self._inflight.get(key)
        if existing is not None:
            return await asyncio.shield(existing)
        task = asyncio.create_task(self._request_json(request, path, key, len(encoded)))
        self._inflight[key] = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self._inflight.pop(key, None)

    async def _request_json(
        self, request: dict[str, Any], path: Path, key: str, request_bytes: int
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        actual_request = dict(request)
        replica_index = 0
        transient_failures = 0
        attempt_outcomes = []
        for attempt in range(self.attempts):
            started = time.time()
            serving_client = self._replicas[replica_index % len(self._replicas)]
            try:
                async with serving_client._semaphore:
                    response = await serving_client._client.chat.completions.create(
                        **actual_request
                    )
                self.stats["requests"] += 1
                usage = response.usage.model_dump() if response.usage else {}
                for field in ("prompt_tokens", "completion_tokens"):
                    self.stats[field] += usage.get(field, 0)
                choice = response.choices[0]
                attempt_outcomes.append({
                    "attempt": attempt + 1, "max_tokens": actual_request["max_tokens"],
                    "finish_reason": choice.finish_reason, "usage": usage,
                    "elapsed_seconds": time.time() - started,
                })
                if choice.finish_reason != "stop":
                    if choice.finish_reason == "length":
                        actual_request["max_tokens"] = min(
                            max(actual_request["max_tokens"] * 2, 1024), 32768
                        )
                    raise InvalidModelJSON(f"Incomplete output: {choice.finish_reason}")
                result = parse_json_object(choice.message.content or "")
                payload = {
                    "cache_key": key,
                    "model": self.runtime["model"],
                    "model_revision": self.runtime.get("revision"),
                    "serving_app_id": serving_client.runtime.get("app_id"),
                    "result": result,
                    "usage": usage,
                    "request_bytes": request_bytes,
                    "messages_sha256": hashlib.sha256(json.dumps(
                        request["messages"], ensure_ascii=False, sort_keys=True
                    ).encode()).hexdigest(),
                    "message_content_characters": [
                        len(message.get("content", ""))
                        if isinstance(message.get("content"), str) else None
                        for message in request["messages"]
                    ],
                    "response_format_sha256": hashlib.sha256(json.dumps(
                        request["response_format"], sort_keys=True
                    ).encode()).hexdigest(),
                    "finish_reason": choice.finish_reason,
                    "requested_max_tokens": request["max_tokens"],
                    "actual_max_tokens": actual_request["max_tokens"],
                    "attempt": attempt + 1,
                    "transient_failures": transient_failures,
                    "attempt_outcomes": attempt_outcomes,
                    "decoding": {
                        name: request.get(name) for name in
                        ("temperature", "top_p", "presence_penalty", "seed")
                    } | request["extra_body"],
                    "elapsed_seconds": time.time() - started,
                    "finished_at_epoch": time.time(),
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".response-")
                try:
                    with os.fdopen(fd, "w") as handle:
                        json.dump(payload, handle, ensure_ascii=False)
                    os.replace(temporary, path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
                return result
            except APIStatusError as exc:
                if exc.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                    # Context overflow and malformed schemas are actionable errors.
                    # Never turn a long-history failure into input truncation.
                    raise
                last_error = exc
                transient_failures += 1
                replica_index += 1
            except (APIConnectionError, APITimeoutError) as exc:
                last_error = exc
                transient_failures += 1
                replica_index += 1
            except InvalidModelJSON as exc:
                # Content failures are not evidence that a replica is unavailable.
                last_error = exc
            if transient_failures >= self.transient_attempts:
                raise RuntimeError(
                    f"Inference unavailable after {transient_failures} transient failures"
                ) from last_error
            if attempt + 1 < self.attempts:
                self.stats["retries"] += 1
                await asyncio.sleep(min(2 ** attempt, 30) + random.random())
        raise RuntimeError(f"JSON inference failed after {self.attempts} attempts") from last_error


class GraphLLMPool:
    """Identical model replicas with stable rollout affinity and a shared cache.

    Use ``pool.shard(rollout_id).complete_json(...)`` to keep every prefix of a
    rollout on the same GPU. Reflection calls can use ``pool.complete_json``.
    No input is split across model contexts: each replica receives the complete
    request. Changing the pool size changes affinity but never cache identity.
    """

    def __init__(
        self,
        runtime_paths: list[str | Path],
        cache_dir: str | Path = "results/cache/graph_llm",
        **kwargs: Any,
    ) -> None:
        if not runtime_paths:
            raise ValueError("At least one model runtime is required")
        configs = [json.loads(Path(path).read_text()) for path in runtime_paths]
        identities = {(config["model"], config.get("revision")) for config in configs}
        if len(identities) != 1:
            raise ValueError("A model pool requires identical models and revisions")
        self.clients = [GraphLLM(path, cache_dir, **kwargs) for path in runtime_paths]
        shared_inflight: dict[str, asyncio.Task[dict[str, Any]]] = {}
        for index, client in enumerate(self.clients):
            client._inflight = shared_inflight
            client._replicas = self.clients[index:] + self.clients[:index]
        self.runtime = self.clients[0].runtime
        self._next = 0

    async def __aenter__(self) -> GraphLLMPool:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await asyncio.gather(*(client.close() for client in self.clients))

    @property
    def stats(self) -> dict[str, int]:
        return {
            key: sum(client.stats[key] for client in self.clients)
            for key in self.clients[0].stats
        }

    def shard(self, route_key: str) -> GraphLLM:
        digest = hashlib.sha256(route_key.encode()).digest()
        return self.clients[int.from_bytes(digest[:8], "big") % len(self.clients)]

    async def token_count(
        self, messages: list[dict[str, Any]], *, thinking: bool = False
    ) -> int:
        return await self.clients[0].token_count(messages, thinking=thinking)

    async def complete_json(
        self, messages: list[dict[str, Any]], *, route_key: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        if route_key is None:
            client = self.clients[self._next % len(self.clients)]
            self._next += 1
        else:
            client = self.shard(route_key)
        return await client.complete_json(messages, **kwargs)


def complete_json_sync(
    messages: list[dict[str, Any]],
    *,
    runtime_path: str | Path = "results/runtime/graph.json",
    cache_dir: str | Path = "results/cache/graph_llm",
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience bridge for synchronous callbacks with infrequent requests."""
    async def run() -> dict[str, Any]:
        async with GraphLLM(runtime_path, cache_dir) as client:
            return await client.complete_json(messages, **kwargs)

    return asyncio.run(run())
