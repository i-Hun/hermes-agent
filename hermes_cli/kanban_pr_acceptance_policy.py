"""Owner-declared acceptance policy for repositories whose GitHub plan hides branch rules.

GitHub Free answers ``rules/branches`` with HTTP 403 "Upgrade to GitHub Pro" for private
repositories, so a repository that gates merges with a self-hosted status (``ci/local``) plus a
SHA-bound review comment has no readable required checks. ``kanban.pr_acceptance.local_policies``
in config.yaml lets the owner declare that policy for ONE repository at a time::

    kanban:
      pr_acceptance:
        local_policies:
          OWNER/REPO:
            required_contexts: [ci/local]      # exact-head statuses/check runs, all must succeed
            review_approve: true               # last "REVIEW: APPROVE" comment pins sha + reviewer-card
            require_merged: true               # PR must be MERGED

The policy only substitutes for plan-gated rule reads; it never turns an arbitrary 403, an API
outage or a missing declaration into "no checks". Unknown keys or wrong types disable the policy.
"""
from __future__ import annotations

import re

_PLAN_GATED = "Upgrade to GitHub Pro or make this repository public"
_VERDICT = re.compile(r"^\s*REVIEW: (APPROVE|REQUEST_CHANGES)\s*(?:\n|$)")
_SHA_LINE = re.compile(r"^\s*sha:\s*([0-9a-f]{40})\s*$", re.M)
_CARD_LINE = re.compile(r"^\s*reviewer-card:\s*(t_[0-9a-f]+)\s*$", re.M)
_AUTHOR_BRANCH = re.compile(r"card/(t_[0-9a-f]+)")
_KEYS = {"required_contexts", "review_approve", "require_merged"}


class PlanGatedRules(Exception):
    """GitHub refused the rules read because of the account plan, not because of auth/outage."""


def is_plan_gated(stdout: str) -> bool:
    return _PLAN_GATED in (stdout or "") and '"403"' in (stdout or "")


def local_policy(repo: str) -> dict | None:
    """Return the validated policy for ``repo`` or None (no/invalid declaration)."""
    try:
        from hermes_cli.config import load_config_readonly
        raw = (((load_config_readonly() or {}).get("kanban") or {}).get("pr_acceptance") or {})
        policy = (raw.get("local_policies") or {}).get(repo)
    except Exception:
        return None
    if not isinstance(policy, dict) or set(policy) - _KEYS:
        return None
    contexts = policy.get("required_contexts")
    if not (isinstance(contexts, list) and contexts and all(isinstance(c, str) and c for c in contexts)):
        return None
    if not all(isinstance(policy.get(k, False), bool) for k in ("review_approve", "require_merged")):
        return None
    return {"required_contexts": list(contexts), "review_approve": bool(policy.get("review_approve")),
            "require_merged": bool(policy.get("require_merged"))}


def review_verdict(comments: list[dict], head_sha: str, head_ref: str) -> tuple[str, dict]:
    """Classify the LAST review verdict comment against the exact head (merge-pr.sh semantics).

    Returns (classification, evidence). Only an APPROVE that names this exact head SHA and a
    reviewer card different from the author card in ``card/<task_id>`` is ``success``.
    """
    verdicts = [c for c in comments if isinstance(c.get("body"), str) and _VERDICT.match(c["body"])]
    if not verdicts:
        return "missing", {"name": "review", "classification": "missing", "head_sha": head_sha}
    last = verdicts[-1]
    body = last["body"]
    verdict = _VERDICT.match(body)
    kind = verdict[1] if verdict else "REQUEST_CHANGES"
    sha = _SHA_LINE.search(body)
    card = _CARD_LINE.search(body)
    author = _AUTHOR_BRANCH.fullmatch(head_ref or "")
    evidence = {"name": "review", "id": last.get("id"), "url": last.get("html_url"),
                "head_sha": sha[1] if sha else None, "verdict": kind,
                "reviewer_card": card[1] if card else None}
    if kind != "APPROVE":
        classification = "failure"
    elif not sha or sha[1] != head_sha:
        classification = "stale"
    elif not card or not author or card[1] == author[1]:
        classification = "missing"
    else:
        classification = "success"
    evidence["classification"] = classification
    return classification, evidence
