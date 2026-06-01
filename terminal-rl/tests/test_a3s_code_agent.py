from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

TERMINAL_RL_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = TERMINAL_RL_DIR.parent
if str(TERMINAL_RL_DIR) not in sys.path:
    sys.path.insert(0, str(TERMINAL_RL_DIR))
if str(ROOT_DIR / "slime") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "slime"))

import agent.a3s_code_agent as a3s_agent_module


class DummyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        _ = add_special_tokens
        return {"input_ids": list(range(len(str(text).split())))}


class DummySGLangClient:
    tokenizer = DummyTokenizer()


def test_a3s_code_agent_run_model_turn_with_mock_sdk(monkeypatch, tmp_path):
    config_path = tmp_path / "a3s-code.hcl"
    config_path.write_text("# fake config\n")

    class FakeResult:
        text = "done"
        tool_calls = [{"name": "read_file"}]
        tool_calls_count = 1
        prompt_tokens = 7
        completion_tokens = 3
        total_tokens = 10
        finish_reason = "stop"
        latency_ms = 12.5

    class FakeSession:
        closed = False

        def send(self, prompt):
            assert prompt == "fix the bug"
            return FakeResult()

        def close(self):
            self.closed = True

    fake_session = FakeSession()

    class FakeAgent:
        @classmethod
        def create(cls, path):
            assert path == str(config_path)
            return cls()

        def session(self, workspace, opts, permissive=True):
            assert Path(workspace).exists()
            assert opts.session_id
            assert permissive is True
            return fake_session

    class FakeSessionOptions:
        pass

    monkeypatch.setattr(
        a3s_agent_module,
        "_bootstrap_a3s_code",
        lambda: (FakeAgent, FakeSessionOptions),
    )
    monkeypatch.setenv("A3S_CODE_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("A3S_CODE_WORKSPACE_ROOT", str(tmp_path / "workspaces"))
    monkeypatch.setenv("A3S_CODE_TURN_TIMEOUT_SEC", "5")

    agent = a3s_agent_module.A3SCodeAgent(
        model_type="Qwen3",
        sglang_client=DummySGLangClient(),
        env_client=None,
        lease_id=None,
        run_context=types.SimpleNamespace(uid="abc123"),
        task_meta={"task_path": "seta_env/1"},
        max_total_tokens=8192,
    )
    agent.start_turn_loop("fix the bug")
    context, terminated = asyncio.run(agent.get_turn_context())
    assert terminated is None
    assert context == [{"role": "user", "content": "fix the bug"}]

    result = asyncio.run(
        agent.run_model_turn(
            context_messages=context,
            sglang_client=DummySGLangClient(),
            tool_schemas=[],
            turn_idx=0,
        )
    )
    assert result.interaction.output_text == "done"
    assert result.interactions == [result.interaction]
    assert result.tool_call_requests == []
    assert result.model_response.tool_calls_count == 1

    final = agent.finalize_response(result.model_response)
    assert isinstance(final, a3s_agent_module.A3SCodeFinalResponse)
    assert final.msg == "done"
    assert final.info["harness_option"] == "a3s-code"

    asyncio.run(agent.close())
    assert fake_session.closed is True
