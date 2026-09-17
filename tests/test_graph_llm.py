"""Inference integrity checks: preserve input and cache only complete outputs."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

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
    runtime_paths = []
    for index in range(2):
        runtime = tmp_path / f"runtime-{index}.json"
        runtime.write_text(json.dumps({
            "api_key": "test-only", "api_base": f"http://localhost:{index + 1}/v1",
            "model": "test-model", "revision": "test-revision",
        }))
        runtime_paths.append(runtime)

    async def run():
        async with GraphLLMPool(runtime_paths, tmp_path / "cache") as pool:
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
