"""Inference integrity checks: preserve input and cache only complete outputs."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import httpx
from openai import APIConnectionError, BadRequestError

from superstate_graphs.graph_llm import GraphLLM, GraphLLMPool, InvalidModelJSON, parse_json_object


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
