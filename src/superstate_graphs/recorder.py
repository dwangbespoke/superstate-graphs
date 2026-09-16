"""Minimal instrumentation of Harbor's stock Terminus-2 policy."""
from __future__ import annotations

import copy
import hashlib
import json
import time

from harbor.agents.terminus_2.terminus_2 import Terminus2


class RecordingTerminus2(Terminus2):
    @staticmethod
    def name() -> str:
        return "superstate-terminus2"

    async def _query_llm(self, chat, prompt, original_instruction="", session=None):
        # This is the chat-level request actually passed to the fixed policy.
        # Outcomes are stored separately by Harbor and never enter this record.
        prefix = copy.deepcopy(chat.messages) + [{"role": "user", "content": prompt}]
        encoded = json.dumps(prefix, ensure_ascii=False, sort_keys=True)
        record = {"schema": "superstate-policy-call-v1", "trial_id": self.logs_dir.parent.name,
                  "call_index": getattr(self, "_sg_call_index", 0),
                  "observed_at_epoch": time.time(), "messages": prefix,
                  "prefix_sha256": hashlib.sha256(encoded.encode()).hexdigest()}
        self._sg_call_index = record["call_index"] + 1
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with (self.logs_dir / "policy_calls.jsonl").open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return await super()._query_llm(chat, prompt, original_instruction, session)
