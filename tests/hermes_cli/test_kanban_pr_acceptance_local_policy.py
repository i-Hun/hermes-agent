"""Owner-declared acceptance for plan-gated private repos, through real SQLite + a `gh` shim.

GitHub Free answers rules/branches with 403 "Upgrade to GitHub Pro" for private repositories.
The declared policy may stand in ONLY for that answer; every other gap must keep the card open.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect

HEAD = "a" * 40
OLD = "b" * 40
PR_URL = "https://github.com/acme/repo/pull/7"
PLAN_GATED = {"message": "Upgrade to GitHub Pro or make this repository public to enable this feature.",
              "documentation_url": "https://docs.github.com/rest/repos/rules#get-rules-for-a-branch",
              "status": "403"}
POLICY = {"required_contexts": ["ci/local"], "review_approve": True, "require_merged": True}


def _approve(sha=HEAD, card="t_beef", kind="APPROVE"):
    return {"id": 9, "html_url": PR_URL + "#issuecomment-9",
            "body": f"REVIEW: {kind}\nsha: {sha}\nreviewer-card: {card}\n\nlooks good"}


@pytest.fixture
def github(tmp_path, monkeypatch):
    state = {}

    def reset(**kw):
        state.clear()
        state.update(rules="plan", status="success", status_present=True, merged=True,
                     comments=[_approve()], comments_fail=False, protection=None)
        state.update(kw)

    reset()
    state_ref = {"reset": reset, "state": state}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, value):
            self.send_response(code)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def do_GET(self):
            path = self.path
            if path == "/graphql":
                return self._send(200, {"data": {"repository": {"pullRequest": {
                    "headRefOid": HEAD, "baseRefName": "master",
                    "state": "MERGED" if state["merged"] else "OPEN",
                    "baseRef": {"branchProtectionRule": state["protection"]}}}}})
            if "/rules/branches/" in path:
                if state["rules"] == "plan":
                    return self._send(403, [PLAN_GATED])
                if state["rules"] == "forbidden":
                    return self._send(403, [{"message": "Resource not accessible by integration", "status": "403"}])
                return self._send(200, [[]])
            if "/check-runs" in path:
                return self._send(200, [{"total_count": 0, "check_runs": []}])
            if "/statuses" in path:
                statuses = [{"id": 55, "context": "ci/local", "state": state["status"],
                             "target_url": "https://example.invalid/ci"}] if state["status_present"] else []
                return self._send(200, [statuses])
            if "/issues/7/comments" in path:
                if state["comments_fail"]:
                    return self._send(502, {"message": "Server Error"})
                return self._send(200, [state["comments"]])
            if "/pulls/7" in path:
                return self._send(200, {"head": {"sha": HEAD, "ref": "card/t_cafe"}, "base": {"ref": "master"},
                                        "state": "closed" if state["merged"] else "open",
                                        "merged": state["merged"],
                                        "merge_commit_sha": "c" * 40 if state["merged"] else None})
            return self._send(404, {"message": "Not Found"})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    # Same contract as real `gh api`: body on stdout, non-zero exit on HTTP errors.
    gh.write_text(f"#!{sys.executable}\nimport sys,urllib.request,urllib.error\n"
                  f"u='http://127.0.0.1:{server.server_port}/'+sys.argv[2]\n"
                  "try:\n    print(urllib.request.urlopen(u).read().decode())\n"
                  "except urllib.error.HTTPError as e:\n    print(e.read().decode()); sys.exit(1)\n")
    gh.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("HERMES_HOME", str(home))

    def declare(policies):
        import yaml
        (home / "config.yaml").write_text(yaml.safe_dump({"kanban": {"pr_acceptance": {"local_policies": policies}}}))

    state_ref["declare"] = declare
    kb.init_db()
    try:
        yield state_ref
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _complete(conn):
    tid = kb.create_task(conn, title="Merge", completion_contract=PR_URL)
    ok = kb.complete_task(conn, tid, result="merged", metadata={"published_pr": PR_URL})
    receipt = json.loads(conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' ORDER BY id DESC",
        (tid,)).fetchone()[0])
    assert (kb.get_task(conn, tid).status == "done") is ok
    return ok, receipt


@pytest.mark.linux_only
def test_declared_local_policy_accepts_exact_head_ci_approve_and_merge(github):
    github["declare"]({"acme/repo": POLICY})
    with connect() as conn:
        ok, receipt = _complete(conn)
    assert ok and receipt["classification"] == "success"
    assert receipt["policy"]["repo"] == "acme/repo"
    names = {c["name"]: c for c in receipt["checks"]}
    assert names["ci/local"]["id"] == 55 and names["ci/local"]["head_sha"] == HEAD
    assert names["review"]["head_sha"] == HEAD and names["review"]["reviewer_card"] == "t_beef"
    assert names["merged"]["merge_commit_sha"] == "c" * 40


@pytest.mark.linux_only
@pytest.mark.parametrize("policies, fault, expected", [
    ({}, {}, "infra"),                                              # plan-gated 403, nothing declared
    ({"other/repo": POLICY}, {}, "infra"),                          # policy is per repository
    ({"acme/repo": {**POLICY, "extra": 1}}, {}, "infra"),           # malformed declaration
    ({"acme/repo": {**POLICY, "required_contexts": []}}, {}, "infra"),
    ({"acme/repo": POLICY}, {"rules": "forbidden"}, "infra"),       # a different 403 is not "no rules"
    ({"acme/repo": POLICY}, {"comments_fail": True}, "infra"),      # API outage
    ({"acme/repo": POLICY}, {"status": "failure"}, "failure"),      # red CI
    ({"acme/repo": POLICY}, {"status": "pending"}, "pending"),
    ({"acme/repo": POLICY}, {"status_present": False}, "missing"),  # no ci/local on this head
    ({"acme/repo": POLICY}, {"comments": []}, "missing"),           # no review verdict
    ({"acme/repo": POLICY}, {"comments": [_approve(), _approve(kind="REQUEST_CHANGES")]}, "failure"),
    ({"acme/repo": POLICY}, {"comments": [_approve(sha=OLD)]}, "stale"),          # approve on old head
    ({"acme/repo": POLICY}, {"comments": [_approve(card="t_cafe")]}, "missing"),  # author approved self
    ({"acme/repo": POLICY}, {"merged": False}, "pending"),          # approved but not merged
    ({"acme/repo": POLICY}, {"protection": {"requiredStatusChecks": []}}, "infra"),
])
def test_every_gap_fails_closed(github, policies, fault, expected):
    github["declare"](policies)
    github["reset"](**fault)
    with connect() as conn:
        ok, receipt = _complete(conn)
    assert not ok
    assert receipt["classification"] == expected
