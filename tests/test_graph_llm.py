"""Inference integrity checks: preserve input and cache only complete outputs."""

import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import httpx
from openai import APIConnectionError, BadRequestError

from superstate_graphs.graph_llm import (
    OUTPUT_RETRY_POLICY, GraphLLM, GraphLLMPool, InvalidModelJSON,
    parse_json_object, qwen_generation_policy,
)


def response(content, finish_reason="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason=finish_reason,
            message=SimpleNamespace(content=content),
        )],
        usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 5, "completion_tokens": 3}),
    )


def make_client(tmp_path):
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({
        "api_key": "test-only", "api_base": "http://localhost:1/v1",
        "model": "test-model", "revision": "test-revision",
    }))
    return GraphLLM(runtime, tmp_path / "cache", attempts=2)


def make_pool(tmp_path, **kwargs):
    runtime_paths = []
    for index in range(2):
        runtime = tmp_path / f"runtime-{index}.json"
        runtime.write_text(json.dumps({
            "api_key": "test-only", "api_base": f"http://localhost:{index + 1}/v1",
            "model": "test-model", "revision": "test-revision", "app_id": f"app-{index}",
        }))
        runtime_paths.append(runtime)
    return GraphLLMPool(runtime_paths, tmp_path / "cache", **kwargs)


def test_full_input_and_model_revision_determine_cache(tmp_path):
    async def run():
        async with make_client(tmp_path) as client:
            call = AsyncMock(return_value=response('{"state":"a"}'))
            client._client.chat.completions.create = call
            long_history = "full-history-" * 10000
            first = [{"role": "user", "content": long_history + "first-ending"}]
            second = [{"role": "user", "content": long_history + "second-ending"}]
            assert await client.complete_json(first) == {"state": "a"}
            assert await client.complete_json(first) == {"state": "a"}
            await client.complete_json(second)
            assert call.await_count == 2
            assert call.call_args_list[0].kwargs["messages"] == first
            assert call.call_args_list[1].kwargs["messages"] == second
            client.runtime["revision"] = "new-revision"
            await client.complete_json(first)
            assert call.await_count == 3
    asyncio.run(run())


def test_thinking_policy_is_bounded_recorded_and_part_of_cache(tmp_path):
    async def run():
        async with make_client(tmp_path) as client:
            call = AsyncMock(return_value=response('{"state":"a"}'))
            client._client.chat.completions.create = call
            messages = [{"role": "user", "content": "complete original history"}]
            await client.complete_json(messages, thinking=True, max_tokens=2048)
            request = call.call_args.kwargs
            assert request["messages"] == messages
            assert request["temperature"] == 0.6
            assert request["top_p"] == 0.95
            assert request["extra_body"]["top_k"] == 20
            assert request["extra_body"]["thinking_token_budget"] == 1024
            assert request["extra_body"]["structured_outputs"] == {
                "json_object": True, "disable_any_whitespace": True,
            }
            await client.complete_json(messages, thinking=True, max_tokens=2048)
            assert call.await_count == 1
            await client.complete_json(messages, thinking=True, max_tokens=2048,
                                       thinking_token_budget=512, temperature=0.2)
            assert call.await_count == 2
            assert call.call_args.kwargs["temperature"] == 0.2
            assert call.call_args.kwargs["extra_body"]["thinking_token_budget"] == 512
            records = [json.loads(p.read_text()) for p in (tmp_path / "cache").glob("*/*.json")]
            assert {r["decoding"]["thinking_token_budget"] for r in records} == {512, 1024}
            assert all(r["attempt_outcomes"][0]["finish_reason"] == "stop" for r in records)
    assert qwen_generation_policy(thinking=False, max_tokens=128) == {
        "temperature": 0.0, "structured_output_policy": {
            "disable_any_whitespace": True, "duplicate_response_format_constraint": True,
        },
        "output_retry_policy": OUTPUT_RETRY_POLICY,
    }
    assert qwen_generation_policy(thinking=True, max_tokens=16384)["thinking_token_budget"] == 4096
    asyncio.run(run())


def test_incomplete_output_retries_without_reducing_input(tmp_path, monkeypatch):
    async def run():
        async with make_client(tmp_path) as client:
            call = AsyncMock(side_effect=[response('{"sta', "length"), response('{"state":"b"}')])
            client._client.chat.completions.create = call
            monkeypatch.setattr("superstate_graphs.graph_llm.asyncio.sleep", AsyncMock())
            messages = [{"role": "user", "content": "unchanged full history"}]
            assert await client.complete_json(messages, max_tokens=128) == {"state": "b"}
            assert call.call_args_list[0].kwargs["messages"] == messages
            assert call.call_args_list[1].kwargs["messages"] == messages
            assert call.call_args_list[1].kwargs["max_tokens"] > 128
            assert [item.kwargs["seed"] for item in call.call_args_list] == [17, 18]
            assert len(list((tmp_path / "cache").glob("*/*.json"))) == 1
    asyncio.run(run())


@pytest.mark.parametrize("value", ['{"state":', "[]", 'prefix {"state":"a"}'])
def test_invalid_json_is_not_silently_repaired(value):
    with pytest.raises(InvalidModelJSON):
        parse_json_object(value)


def test_pool_shards_consistently_and_deduplicates_concurrent_requests(tmp_path):
    async def run():
        async with make_pool(tmp_path) as pool:
            calls = [AsyncMock(return_value=response('{"state":"a"}')) for _ in range(2)]
            for client, call in zip(pool.clients, calls):
                client._client.chat.completions.create = call
            assert pool.shard("rollout-17") is pool.shard("rollout-17")
            messages = [{"role": "user", "content": "same complete history"}]
            results = await asyncio.gather(
                pool.complete_json(messages), pool.complete_json(messages)
            )
            assert results == [{"state": "a"}, {"state": "a"}]
            assert sum(call.await_count for call in calls) == 1
    asyncio.run(run())


def test_sharded_client_fails_over_only_after_transient_error(tmp_path, monkeypatch):
    async def run():
        async with make_pool(tmp_path) as pool:
            unavailable = APIConnectionError(request=httpx.Request("POST", "http://primary/v1"))
            calls = [AsyncMock(side_effect=unavailable),
                     AsyncMock(return_value=response('{"state":"b"}'))]
            for client, call in zip(pool.clients, calls):
                client._client.chat.completions.create = call
            monkeypatch.setattr("superstate_graphs.graph_llm.asyncio.sleep", AsyncMock())
            messages = [{"role": "user", "content": "complete original history"}]
            result = await pool.clients[0].complete_json(messages)
            assert result == {"state": "b"}
            assert [call.await_count for call in calls] == [1, 1]
            assert calls[1].call_args.kwargs["messages"] == messages
            record = json.loads(next((tmp_path / "cache").glob("*/*.json")).read_text())
            assert record["serving_app_id"] == "app-1"
            assert record["transient_failures"] == 1
            assert record["model_revision"] == "test-revision"
    asyncio.run(run())


def test_pool_does_not_fail_over_for_invalid_schema(tmp_path):
    async def run():
        async with make_pool(tmp_path) as pool:
            invalid = BadRequestError(
                "Invalid schema",
                response=httpx.Response(400, request=httpx.Request("POST", "http://primary/v1")),
                body={"error": "invalid schema"},
            )
            calls = [AsyncMock(side_effect=invalid), AsyncMock()]
            for client, call in zip(pool.clients, calls):
                client._client.chat.completions.create = call
            with pytest.raises(BadRequestError):
                await pool.clients[0].complete_json([{"role": "user", "content": "history"}])
            assert [call.await_count for call in calls] == [1, 0]
    asyncio.run(run())


def test_transient_failure_budget_is_bounded(tmp_path, monkeypatch):
    async def run():
        async with make_pool(tmp_path, attempts=6, transient_attempts=3) as pool:
            unavailable = APIConnectionError(request=httpx.Request("POST", "http://primary/v1"))
            calls = [AsyncMock(side_effect=unavailable), AsyncMock(side_effect=unavailable)]
            for client, call in zip(pool.clients, calls):
                client._client.chat.completions.create = call
            monkeypatch.setattr("superstate_graphs.graph_llm.asyncio.sleep", AsyncMock())
            with pytest.raises(RuntimeError, match="3 transient failures"):
                await pool.clients[0].complete_json([{"role": "user", "content": "history"}])
            assert sum(call.await_count for call in calls) == 3
            assert not list((tmp_path / "cache").glob("*/*.json"))
    asyncio.run(run())


def test_output_retries_change_seed_but_transients_preserve_it_and_inputs(tmp_path, monkeypatch):
    async def run():
        async with make_client(tmp_path) as client:
            client.attempts = 6
            unfinished = response('test-only' + 'x' * 12000, "length")
            unfinished.choices[0].message.reasoning = 'test-only' + 'r' * 12000
            unavailable = APIConnectionError(request=httpx.Request("POST", "http://primary/v1"))
            call = AsyncMock(side_effect=[
                unfinished, unavailable, response('{"st', "length"),
                response('not-json'), response('{"state":"recovered"}'),
            ])
            client._client.chat.completions.create = call
            monkeypatch.setattr("superstate_graphs.graph_llm.asyncio.sleep", AsyncMock())
            messages = [{"role": "user", "content": "PRIVATE_FULL_PREFIX_" * 10000}]
            schema = {"type": "object", "properties": {"state": {"type": "string"}}}
            assert await client.complete_json(
                messages, schema=schema, thinking=True, max_tokens=2048
            ) == {"state": "recovered"}
            requests = [item.kwargs for item in call.call_args_list]
            assert [r["seed"] for r in requests] == [17, 18, 18, 19, 20]
            assert [r["max_tokens"] for r in requests] == [2048, 4096, 4096, 8192, 8192]
            assert all(r["messages"] == messages for r in requests)
            assert all(r["response_format"] == requests[0]["response_format"] for r in requests)
            assert all(r["extra_body"] == requests[0]["extra_body"] for r in requests)
            records = list((tmp_path / "cache").glob("*/*.json"))
            assert len(records) == 1
            result = json.loads(records[0].read_text())
            assert result["requested_seed"] == 17 and result["actual_seed"] == 20
            assert result["decoding"]["seed"] == 20
            assert result["invalid_outputs"] == 3 and result["transient_failures"] == 1
            assert [r["seed"] for r in result["attempt_outcomes"]] == [17, 18, 18, 19, 20]
            failures = list((tmp_path / "cache_failed_responses").glob("*/*.json"))
            assert len(failures) == 3
            for path in failures:
                raw = path.read_text()
                failed = json.loads(raw)
                assert len(failed["content_preview"]) + len(failed["reasoning_preview"]) <= 8192
                assert all(secret not in raw for secret in (
                    "test-only", "PRIVATE_FULL_PREFIX_", "api_key", "api_base", '"messages"'
                ))
                assert len(failed["request_sha256"]) == 64
                assert failed["seed"] in {17, 18, 19}
            assert await client.complete_json(
                messages, schema=schema, thinking=True, max_tokens=2048
            ) == {"state": "recovered"}
            assert call.await_count == 5
    asyncio.run(run())


def test_retry_policy_invalidates_base_cache_and_excludes_legacy_identity(tmp_path, monkeypatch):
    async def run():
        async with make_client(tmp_path) as client:
            call = AsyncMock(return_value=response('{"state":"new"}'))
            client._client.chat.completions.create = call
            messages = [{"role": "user", "content": "same full input"}]
            await client.complete_json(messages)
            request = call.call_args.kwargs
            next((tmp_path / "cache").glob("*/*.json")).unlink()
            legacy_key = hashlib.sha256(json.dumps({
                "namespace": "graph-v1", "model_revision": "test-revision", "request": request,
            }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            legacy = tmp_path / "cache" / legacy_key[:2] / f"{legacy_key}.json"
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_text(json.dumps({"result": {"state": "legacy"}}))
            assert await client.complete_json(messages) == {"state": "new"}
            assert call.await_count == 2
            previous = qwen_generation_policy(thinking=False, max_tokens=2048)
            monkeypatch.setitem(OUTPUT_RETRY_POLICY, "version", "test-distinct-policy")
            assert qwen_generation_policy(thinking=False, max_tokens=2048) != previous
            await client.complete_json(messages)
            assert call.await_count == 3
    asyncio.run(run())


def test_exhausted_invalid_outputs_keep_diagnostics_without_success_cache(tmp_path, monkeypatch):
    async def run():
        async with make_client(tmp_path) as client:
            call = AsyncMock(return_value=response('{"bad', "length"))
            client._client.chat.completions.create = call
            monkeypatch.setattr("superstate_graphs.graph_llm.asyncio.sleep", AsyncMock())
            with pytest.raises(RuntimeError, match="after 2 attempts"):
                await client.complete_json([{"role": "user", "content": "full history"}])
            assert [r.kwargs["seed"] for r in call.call_args_list] == [17, 18]
            assert not list((tmp_path / "cache").glob("*/*.json"))
            assert len(list((tmp_path / "cache_failed_responses").glob("*/*.json"))) == 2
    asyncio.run(run())
