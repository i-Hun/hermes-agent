"""Per-profile extra MCP servers reach the engine on ``session/new``.

Local overlay test (not upstream).

Why the feature exists at all: the bridge builds ``mcpServers`` itself
(``copilot_acp_client.__init__``), so ``hermes mcp add`` never reaches the
engine, and a ``.mcp.json`` dropped in a repo is read from the *session cwd* —
which for a Kanban worker is an empty scratch dir, i.e. it would cover the
interactive topic and miss exactly the sessions that write code. Declaring the
servers in the profile's ``config.yaml`` covers both, and stays out of ``.env``
(held read-only by the credential guard) so no operator grant is needed.

The last test drives the real ``_run_prompt`` against a scripted bridge and
asserts on the ``session/new`` params the engine actually receives — the unit
tests above it can only prove the list was *built* right.
"""

from __future__ import annotations

import json
import os
import queue
import tempfile
import textwrap
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import acp_extra_mcp
from agent.acp_extra_mcp import extra_mcp_servers
from agent.copilot_acp_client import CopilotACPClient


def _write_config(root: Path, body: str) -> Path:
    path = root / "config.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


class DeclarationTests(unittest.TestCase):
    """Reading ``acp.mcp_servers`` out of a profile config."""

    def setUp(self) -> None:
        acp_extra_mcp._CACHE.clear()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _servers(self, body: str) -> list[dict]:
        return extra_mcp_servers(path=_write_config(self.root, body))

    def test_missing_config_is_not_an_error(self) -> None:
        self.assertEqual(extra_mcp_servers(path=self.root / "nope.yaml"), [])

    def test_config_without_the_block_yields_nothing(self) -> None:
        self.assertEqual(self._servers("model:\n  default: claude-sonnet-5\n"), [])

    def test_unreadable_yaml_costs_the_servers_not_the_turn(self) -> None:
        """A typo must never raise: this runs while building a live client."""
        self.assertEqual(self._servers("acp: [oops\n  - broken: ["), [])

    def test_mapping_form_becomes_an_acp_http_entry(self) -> None:
        servers = self._servers(
            """
            acp:
              mcp_servers:
                nuxt:
                  type: http
                  url: https://nuxt.com/mcp
            """
        )
        self.assertEqual(
            servers,
            [
                {
                    "name": "nuxt",
                    "type": "http",
                    "url": "https://nuxt.com/mcp",
                    "headers": [],
                }
            ],
        )

    def test_http_entry_always_carries_headers(self) -> None:
        """``headers`` is required by the wire schema even when it is empty.

        ACP validates ``session/new.mcpServers`` with ``vecSkipError(zMcpServer)``
        (@agentclientprotocol/sdk schema/zod.gen.js): an entry that fails the
        union is dropped from the array in silence — session/new still returns a
        sessionId, and the server simply never exists. ``zMcpServerHttp`` lists
        ``headers`` as required, so omitting it costs the whole server with no
        error anywhere. Proven live against the engine on 02.08.2026.
        """
        for kind in ("http", "sse"):
            with self.subTest(kind=kind):
                servers = self._servers(
                    f"""
                    acp:
                      mcp_servers:
                        remote:
                          type: {kind}
                          url: https://example.test/mcp
                    """
                )
                self.assertEqual(servers[0]["headers"], [])

    def test_stdio_entry_always_carries_args_and_env(self) -> None:
        """Same silent-skip trap: ``zMcpServerStdio`` requires both keys."""
        servers = self._servers(
            """
            acp:
              mcp_servers:
                local:
                  command: /usr/bin/thing
            """
        )
        self.assertEqual(servers[0]["args"], [])
        self.assertEqual(servers[0]["env"], [])

    def test_type_is_inferred_from_url(self) -> None:
        servers = self._servers(
            """
            acp:
              mcp_servers:
                nuxt:
                  url: https://nuxt.com/mcp
            """
        )
        self.assertEqual(servers[0]["type"], "http")

    def test_list_form_carries_its_own_name(self) -> None:
        servers = self._servers(
            """
            acp:
              mcp_servers:
                - name: nuxt
                  type: sse
                  url: https://nuxt.com/sse
            """
        )
        self.assertEqual(servers[0]["name"], "nuxt")
        self.assertEqual(servers[0]["type"], "sse")

    def test_headers_mapping_becomes_name_value_pairs(self) -> None:
        """ACP wants ``[{name, value}]``; nobody writes yaml that way."""
        servers = self._servers(
            """
            acp:
              mcp_servers:
                private:
                  url: https://example.test/mcp
                  headers:
                    Authorization: Bearer t0ken
            """
        )
        self.assertEqual(
            servers[0]["headers"], [{"name": "Authorization", "value": "Bearer t0ken"}]
        )

    def test_disabled_entry_is_kept_in_the_file_and_skipped(self) -> None:
        self.assertEqual(
            self._servers(
                """
                acp:
                  mcp_servers:
                    nuxt:
                      url: https://nuxt.com/mcp
                      enabled: false
                """
            ),
            [],
        )

    def test_the_reserved_name_cannot_be_taken(self) -> None:
        """The engine keys servers by name — a clash would REPLACE Hermes' tools."""
        self.assertEqual(
            self._servers(
                """
                acp:
                  mcp_servers:
                    hermes-tools:
                      url: https://evil.test/mcp
                """
            ),
            [],
        )

    def test_a_non_http_url_is_refused(self) -> None:
        self.assertEqual(
            self._servers(
                """
                acp:
                  mcp_servers:
                    weird:
                      type: http
                      url: file:///etc/passwd
                """
            ),
            [],
        )

    def test_unknown_type_is_refused(self) -> None:
        self.assertEqual(
            self._servers(
                """
                acp:
                  mcp_servers:
                    weird:
                      type: carrier-pigeon
                      url: https://example.test/mcp
                """
            ),
            [],
        )

    def test_stdio_entry_carries_no_type_key(self) -> None:
        """acp-agent.js routes stdio through ``!("type" in server)``.

        An explicit ``type: "stdio"`` matches neither branch there and the
        server is dropped silently, so the entry we emit must NOT carry one.
        """
        servers = self._servers(
            """
            acp:
              mcp_servers:
                local:
                  type: stdio
                  command: /usr/bin/thing
                  args: ["--mcp"]
                  env:
                    TOKEN: abc
            """
        )
        self.assertNotIn("type", servers[0])
        self.assertEqual(servers[0]["command"], "/usr/bin/thing")
        self.assertEqual(servers[0]["args"], ["--mcp"])
        self.assertEqual(servers[0]["env"], [{"name": "TOKEN", "value": "abc"}])

    def test_one_bad_entry_does_not_take_the_good_ones_down(self) -> None:
        servers = self._servers(
            """
            acp:
              mcp_servers:
                good:
                  url: https://good.test/mcp
                bad:
                  type: nonsense
                also_good:
                  url: https://also.test/mcp
            """
        )
        self.assertEqual([s["name"] for s in servers], ["good", "also_good"])

    def test_duplicate_names_keep_the_first(self) -> None:
        servers = self._servers(
            """
            acp:
              mcp_servers:
                - name: dup
                  url: https://one.test/mcp
                - name: dup
                  url: https://two.test/mcp
            """
        )
        self.assertEqual(len(servers), 1)
        self.assertEqual(servers[0]["url"], "https://one.test/mcp")

    def test_kill_switch_silences_everything(self) -> None:
        path = _write_config(
            self.root,
            """
            acp:
              mcp_servers:
                nuxt:
                  url: https://nuxt.com/mcp
            """,
        )
        with patch.dict(os.environ, {acp_extra_mcp.ENABLE_VAR: "0"}):
            self.assertEqual(extra_mcp_servers(path=path), [])
        self.assertEqual(len(extra_mcp_servers(path=path)), 1)

    def test_an_edit_is_picked_up_without_a_restart(self) -> None:
        """The cache is keyed by mtime, so a fixed typo takes effect next session."""
        path = _write_config(
            self.root,
            """
            acp:
              mcp_servers:
                nuxt:
                  type: carrier-pigeon
                  url: https://nuxt.com/mcp
            """,
        )
        self.assertEqual(extra_mcp_servers(path=path), [])
        os.utime(path, (0, 0))  # force a distinct cache key
        _write_config(
            self.root,
            """
            acp:
              mcp_servers:
                nuxt:
                  type: http
                  url: https://nuxt.com/mcp
            """,
        )
        self.assertEqual(len(extra_mcp_servers(path=path)), 1)

    def test_callers_cannot_mutate_the_cache(self) -> None:
        path = _write_config(
            self.root,
            """
            acp:
              mcp_servers:
                nuxt:
                  url: https://nuxt.com/mcp
            """,
        )
        extra_mcp_servers(path=path)[0]["url"] = "https://hijacked.test"
        self.assertEqual(extra_mcp_servers(path=path)[0]["url"], "https://nuxt.com/mcp")


# ---------------------------------------------------------------------------
# End-to-end: what the engine is actually handed.
# ---------------------------------------------------------------------------


class _FakeStdout:
    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()

    def emit(self, obj: dict) -> None:
        self._q.put(json.dumps(obj) + "\n")

    def eof(self) -> None:
        self._q.put(None)

    def __iter__(self) -> "_FakeStdout":
        return self

    def __next__(self) -> str:
        item = self._q.get()
        if item is None:
            raise StopIteration
        return item

    def close(self) -> None:
        pass


class _FakeStderr:
    def __iter__(self):
        return iter(())

    def close(self) -> None:
        pass


class _RecordingBridge:
    """A fake ``claude-agent-acp`` that keeps the ``session/new`` params."""

    def __init__(self) -> None:
        self.stdout = _FakeStdout()
        self.stderr = _FakeStderr()
        self.stdin = self
        self.returncode = None
        self.session_new_params: dict | None = None
        self._threads: list[threading.Thread] = []

    def write(self, data: str) -> None:
        for line in data.splitlines():
            if line.strip():
                self._dispatch(json.loads(line))

    def flush(self) -> None:
        pass

    def _dispatch(self, msg: dict) -> None:
        method = msg.get("method")
        req_id = msg.get("id")
        if method == "initialize":
            self.stdout.emit({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif method == "session/new":
            self.session_new_params = msg.get("params") or {}
            self.stdout.emit(
                {"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": "s-1"}}
            )
        elif method == "session/prompt":
            t = threading.Thread(target=self._answer, args=(req_id,), daemon=True)
            t.start()
            self._threads.append(t)

    def _answer(self, req_id) -> None:
        self.stdout.emit(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "messageId": "m-1",
                        "content": {"type": "text", "text": "ok"},
                    }
                },
            }
        )
        self.stdout.emit(
            {"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn"}}
        )

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0
        self.stdout.eof()

    def kill(self) -> None:
        self.terminate()

    def wait(self, timeout=None):
        return self.returncode


class SessionNewTests(unittest.TestCase):
    """The list the client builds is the list the bridge receives."""

    def setUp(self) -> None:
        acp_extra_mcp._CACHE.clear()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        _write_config(
            self.root,
            """
            acp:
              mcp_servers:
                nuxt:
                  type: http
                  url: https://nuxt.com/mcp
            """,
        )

    def _client(self) -> CopilotACPClient:
        return CopilotACPClient(
            api_key="copilot-acp",
            base_url="acp://copilot",
            acp_command="copilot",
            acp_args=["--acp", "--stdio"],
            acp_cwd=str(self.root),
        )

    def test_declared_server_reaches_session_new(self) -> None:
        bridge = _RecordingBridge()
        with patch.dict(os.environ, {"HERMES_HOME": str(self.root)}):
            with patch("agent.acp_native_tools._acp_native_tools_enabled", return_value=False):
                client = self._client()
            with patch("agent.copilot_acp_client.subprocess.Popen", return_value=bridge):
                client._run_prompt("hi", timeout_seconds=10)
        assert bridge.session_new_params is not None
        names = [s["name"] for s in bridge.session_new_params["mcpServers"]]
        self.assertIn("nuxt", names)

    def test_hermes_tools_stays_first(self) -> None:
        """Order is the cheap half of the anti-shadowing guarantee."""
        with patch.dict(os.environ, {"HERMES_HOME": str(self.root)}):
            with patch(
                "agent.copilot_acp_client._acp_native_tools_enabled", return_value=True
            ):
                client = self._client()
        names = [s["name"] for s in client._mcp_servers]
        self.assertEqual(names[0], "hermes-tools")
        self.assertIn("nuxt", names)

    def test_no_declaration_means_the_list_is_unchanged(self) -> None:
        """Every other profile must pay nothing for this feature."""
        empty = Path(tempfile.mkdtemp(dir=self.root))
        with patch.dict(os.environ, {"HERMES_HOME": str(empty)}):
            with patch(
                "agent.copilot_acp_client._acp_native_tools_enabled", return_value=True
            ):
                client = self._client()
        self.assertEqual([s["name"] for s in client._mcp_servers], ["hermes-tools"])


if __name__ == "__main__":
    unittest.main()
