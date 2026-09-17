"""Check actual server launch arguments without starting a Modal container."""

import ast
import json
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


SERVER = Path(__file__).resolve().parents[1] / 'scripts' / 'modal_graph_models.py'


def test_server_enforces_compact_xgrammar_at_engine_startup(monkeypatch):
    # Execute the real start method in isolation; importing the entire Modal app
    # would create app/image/volume objects unrelated to this launch contract.
    tree = ast.parse(SERVER.read_text())
    config_assignment = next(
        node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'STRUCTURED_OUTPUTS_CONFIG'
                for target in node.targets)
    )
    server_class = next(node for node in tree.body
                        if isinstance(node, ast.ClassDef) and node.name == 'GraphModelServer')
    start = next(node for node in server_class.body
                 if isinstance(node, ast.FunctionDef) and node.name == 'start')
    start.decorator_list = []
    module = ast.Module(body=[config_assignment, start], type_ignores=[])
    namespace = {
        'json': json, 'os': os, 'MODEL_ID': 'pinned-model', 'REVISION': 'pinned-revision',
        'TENSOR_PARALLEL': 2, 'MAX_MODEL_LEN': 262144, 'MAX_SEQS': 16,
    }
    process = SimpleNamespace(stdout=iter([]))
    popen = Mock(return_value=process)
    monkeypatch.setattr(subprocess, 'Popen', popen)
    monkeypatch.setattr(threading, 'Thread', Mock())
    monkeypatch.setenv('VLLM_API_KEY', 'test-key')
    monkeypatch.setenv('OMP_NUM_THREADS', '16')
    exec(compile(module, str(SERVER), 'exec'), namespace)
    instance = SimpleNamespace()
    namespace['start'](instance)
    args = popen.call_args.args[0]
    config = json.loads(args[args.index('--structured-outputs-config') + 1])
    assert config == {'backend': 'xgrammar', 'disable_any_whitespace': True}
    assert args[args.index('--reasoning-parser') + 1] == 'qwen3'
    assert args[args.index('--revision') + 1] == 'pinned-revision'
    assert args[args.index('--max-model-len') + 1] == '262144'
    assert args[args.index('--tensor-parallel-size') + 1] == '2'
    assert args[args.index('--max-num-seqs') + 1] == '16'
    assert popen.call_args.kwargs['env']['OMP_NUM_THREADS'] == '1'
    assert instance.process is process
