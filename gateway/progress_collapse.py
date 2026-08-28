"""Collapsible tool-progress history + live subagent transcript tails.

Local overlay module (ACP-native fleet). In "replace" tool-progress grouping
on platforms with expandable blockquotes (Telegram MarkdownV2 ``**> …||``),
older progress lines fold into a collapsed quote instead of scrolling away,
every 💬 narration line of the turn accumulates pinned above the current tool
line, and while ``delegate_task`` children run, the tail of their live transcripts
(#67479, ``cache/delegation/live/<id>/task-N.log``) is surfaced beneath it.

All feature logic lives here on purpose: ``gateway/run.py`` is one of the
most actively changed upstream files, so it keeps only a few thin hook lines
into this module — the conflict surface of a ``hermes update`` stays small.
"""

from __future__ import annotations

import re
import time


class ProgressCollapser:
    """Per-run state + rendering for the progress bubble of one agent turn.

    Instantiated by ``send_progress_messages`` in ``gateway/run.py`` once per
    run; not thread-safe (it is only touched from the progress task).
    """

    POLL_INTERVAL = 1.0        # min seconds between transcript filesystem polls
    HISTORY_LINE_CAP = 120     # flattened history line length
    # 💬 narration is the one progress line that carries meaning, so it is
    # kept whole; the caps below are safety margins, not editorial limits.
    NARRATION_CAP_COLLAPSED = 3500  # 💬 pinned visible — effectively uncapped
    NARRATION_CAP_PLAIN = 500       # 💬 cap in plain rolling-window mode
    NARRATION_HISTORY_CAP = 500     # 💬 in history (plain mode / legacy entries)
    BUBBLE_CAP = 3800          # whole bubble, under Telegram's 4096 limit
    TAIL_LINE_CAP = 160        # 📟 subagent tail line length
    STALE_LOG_SECONDS = 300    # transcript quiet this long → not live
    FINAL_LINGER_SECONDS = 15  # closing "final" line lingers as ✅ feedback

    def __init__(self, grouping: str, adapter) -> None:
        self.replace_mode = grouping == "replace"
        self.collapse_history = self.replace_mode and bool(
            getattr(adapter, "supports_expandable_quote", False)
        )
        self.window = 8 if self.collapse_history else 4
        # Every 💬 of the turn, oldest first. None of them folds into the
        # collapsed history: together they are the readable thread of what the
        # agent said it was doing, and burying all but the newest under the cut
        # hid it (Oleg, 17.08.2026).
        self.pinned_narrations: list = []
        self._last_narration: str | None = None    # for extension detection
        self.subagent_tail: list = []  # rendered live-transcript lines (📟 …)
        self._subagent_sig = ""        # change signature of the tail
        self._subagent_poll_ts = 0.0   # last filesystem poll (monotonic)

    # ── progress-line intake ────────────────────────────────────────────


    # Progress lines read as ACTIONS, the OpenClaw way ("🛠️ check JS syntax
    # for /tmp/app.js"), not as tool plumbing. The gateway's generic renderer
    # wraps a preview as `⚙️ Running <text>` or `⚙️ <tool>: "<text>"`; for
    # engine-driven work the wrapper is noise in front of an already
    # self-describing sentence, so it is stripped here — in the overlay,
    # leaving the upstream renderer untouched for every other caller.
    _TOOL_LINE_RE = re.compile(
        r'^(?P<emoji>\W{1,3})\s*(?:Running\s+|(?P<tool>[\w.-]+)\s*:\s*)(?P<rest>.+)$',
        re.DOTALL,
    )

    @classmethod
    def tidy_tool_line(cls, msg):
        """Strip the renderer's tool wrapper from an engine activity line."""
        if not isinstance(msg, str):
            return msg
        line = msg.strip()
        if not line or line.startswith("💬"):
            return msg
        if "```" in line:
            # Fenced command block from the upstream terminal renderer: keep
            # the header emoji and name the program, drop the argv. Same rule
            # as the ACP path — progress is a public side channel, and a full
            # command line leaks paths/hosts into the chat.
            head, _, body = line.partition("```")
            emoji = (head.strip().split() or ["💻"])[0]
            cmd = body.strip().strip("`").strip().splitlines()
            first = cmd[0].strip() if cmd else ""
            from agent.acp_native_tools import _program_name

            return f"{emoji} {_program_name(first)}" if first else msg
        m = cls._TOOL_LINE_RE.match(line)
        if not m:
            return msg
        rest = m.group("rest").strip()
        if len(rest) > 1 and rest[0] == rest[-1] == '"':
            rest = rest[1:-1].strip()
        if not rest:
            return msg
        return f"{m.group('emoji').strip()} {rest}"

    @staticmethod
    def _narration_extends(prev, new) -> bool:
        """True when ``new`` is the same 💬 narration, grown.

        The engine can surface a tool call while the assistant's text block is
        still streaming (permission path, async tool_call_update), so the live
        pulse for one narration may arrive first partial and then whole. The
        partial is not a separate step — it must be replaced, not stacked, or
        one sentence reads as two entries split mid-word.
        """
        if not isinstance(prev, str) or not isinstance(new, str):
            return False
        a = prev.lstrip()
        b = new.lstrip()
        if not a.startswith("💬") or not b.startswith("💬"):
            return False
        a = a[1:].strip().rstrip("…").strip()
        b = b[1:].strip()
        return bool(a) and b != a and b.startswith(a)

    def append(self, lines: list, msg) -> list:
        """Add one progress entry; in "replace" mode keep a rolling window.

        "replace" — OpenClaw-style progress draft: a rolling window of the
        most recent activity lines (thoughts + tools) in ONE edited bubble.
        Completed work scrolls away instead of accumulating; thought lines
        are flattened and capped so repeated edits don't reflow the bubble
        height (mirrors OpenClaw's progress-draft maxLines/maxLineChars).
        """
        if not self.replace_mode:
            lines.append(msg)
            return lines
        line = self.tidy_tool_line(msg)
        if isinstance(line, str) and line.lstrip().startswith("💬"):
            flat = " ".join(line.split())
            # Narration is the most informative progress line — in collapse
            # mode keep it near-whole (it gets pinned visible); elsewhere
            # keep the tight cap so edits don't reflow the bubble.
            cap = (
                self.NARRATION_CAP_COLLAPSED
                if self.collapse_history
                else self.NARRATION_CAP_PLAIN
            )
            if len(flat) > cap:
                flat = flat[: cap - 1] + "…"
            line = flat
            prev = self._last_narration
            if self._narration_extends(prev, line):
                # Supersede the partial in place: drop it from the window and
                # from the pinned thread, and let the grown line take its slot.
                # One sentence that arrived in two pieces is one entry, not two.
                lines = [e for e in lines if str(e) != prev]
                self.pinned_narrations = [
                    p for p in self.pinned_narrations if p != prev
                ]
            self._last_narration = line
            if self.collapse_history and (
                not self.pinned_narrations or self.pinned_narrations[-1] != line
            ):
                self.pinned_narrations.append(line)
        lines.append(line)
        if len(lines) > self.window:
            lines = lines[-self.window:]
        return lines

    def on_content_bubble(self) -> None:
        """A content bubble landed — the narration it belonged to is spent,
        and the subagent tail must re-render into the NEXT bubble (its
        content didn't change, but its host message is gone)."""
        self.pinned_narrations = []
        self._last_narration = None
        self._subagent_sig = ""

    # ── rendering ───────────────────────────────────────────────────────

    @classmethod
    def flatten_entry(cls, entry) -> str:
        """One-line form of a progress entry for the collapsed history.

        Terminal entries arrive as fenced blocks ("💻 terminal\\n```…```");
        inside a blockquote a fence won't render, so history flattens them
        to "💻 <cmd>" single lines.
        """
        s = str(entry)
        if "```" in s:
            lines = [ln for ln in s.splitlines() if not ln.strip().startswith("```")]
            if lines and lines[0].rstrip().endswith("terminal"):
                head = lines[0].split()
                emoji = head[0] if head else "💻"
                s = f"{emoji} " + " ".join(" ".join(lines[1:]).split())
            else:
                s = " ".join(" ".join(lines).split())
        else:
            s = " ".join(s.split())
        # A superseded 💬 still says what the agent was doing — give it a far
        # looser cap than tool plumbing before it folds into the quote.
        cap = cls.NARRATION_HISTORY_CAP if s.startswith("💬") else cls.HISTORY_LINE_CAP
        if len(s) > cap:
            s = s[: cap - 1] + "…"
        return s

    def render(self, lines: list) -> str:
        """Bubble text: [collapsed history] + pinned 💬 thread + current + tail."""
        if not self.collapse_history:
            text = "\n".join(str(line) for line in lines)
            if self.subagent_tail:
                text = (text + "\n" if text else "") + "\n".join(self.subagent_tail)
            return text
        # 💬 narration outranks the tool lines around it: the whole thread of
        # narrations accumulates visible above the current line, in the order
        # it was said, and none of it ever folds into the cut. Held in a
        # separate list, so entries survive scrolling out of the window.
        current = str(lines[-1]) if lines else ""
        pinned = [p for p in self.pinned_narrations if p != current]
        pinned_set = set(pinned)
        hist_src = [e for e in lines[:-1] if str(e) not in pinned_set]
        history = [self.flatten_entry(e) for e in hist_src]

        def _assemble(hist: list, pins: list) -> str:
            parts: list = []
            if hist:
                # No space after the marker: in Telegram everything past ">"
                # is quote content, so "> x" indents every line but the first
                # (the first loses its space to the Bot API's message-level
                # trim) — the ragged left edge Oleg reported 2026-08-06.
                quoted = [f"**>{hist[0]}"] + [f">{h}" for h in hist[1:]]
                quoted[-1] += "||"
                parts.append("\n".join(quoted))
            parts.extend(pins)
            if lines:
                parts.append(current)
            if self.subagent_tail:
                parts.extend(self.subagent_tail)
            return "\n".join(parts)

        # Narration is neither trimmed to a tight cap nor capped in count, so
        # the bubble can outgrow Telegram's 4096-char message limit and make
        # every edit fail. Shed the oldest history lines first — they are the
        # least informative part — then the oldest narrations, always keeping
        # the newest one; clip the whole bubble only as a last resort.
        text = _assemble(history, pinned)
        while len(text) > self.BUBBLE_CAP and history:
            history = history[1:]
            text = _assemble(history, pinned)
        while len(text) > self.BUBBLE_CAP and len(pinned) > 1:
            pinned = pinned[1:]
            text = _assemble(history, pinned)
        if len(text) > self.BUBBLE_CAP:
            text = text[: self.BUBBLE_CAP - 1] + "…"
        return text

    # ── live subagent transcripts (#67479) ──────────────────────────────

    @staticmethod
    def _last_transcript_line(path):
        """(role, text) of the last event line of a live transcript log."""
        try:
            with open(path, "rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                fh.seek(max(0, size - 4096))
                chunk = fh.read().decode("utf-8", errors="replace")
        except OSError:
            return None
        for raw in reversed(chunk.splitlines()):
            s = raw.strip()
            # Skip blanks and the pre-created header block.
            if not s or s.startswith("=") or s.startswith("("):
                continue
            # Event lines: "HH:MM:SS role     | text"
            if len(s) > 9 and s[2] == ":" and s[5] == ":" and "|" in s:
                head, _, text = s.partition("|")
                role = head[8:].strip() or "…"
                text = " ".join(text.split())
                if text:
                    return role, text
                continue
            return "…", " ".join(s.split())
        return None

    def collect_subagent_tail(self) -> list:
        """Newest line of each live delegation transcript (#67479).

        Purely filesystem-based, so it works however delegate_task was
        invoked (in-process, ACP/MCP bridge, background batch). A task is
        shown while its log stays fresh; the closing "final" line lingers
        ~15s as completion feedback, then drops out.
        """
        try:
            from tools.delegation_live_log import live_transcript_root

            root = live_transcript_root()
            if not root.is_dir():
                return []
            now = time.time()
            dirs = sorted(
                (p for p in root.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:2]
            out: list = []
            for d in dirs:
                for logf in sorted(d.glob("task-*.log"))[:3]:
                    try:
                        st = logf.stat()
                    except OSError:
                        continue
                    if now - st.st_mtime > self.STALE_LOG_SECONDS:
                        continue  # quiet too long → not live
                    parsed = self._last_transcript_line(logf)
                    if parsed is None:
                        continue
                    role, text = parsed
                    if role == "final" and now - st.st_mtime > self.FINAL_LINGER_SECONDS:
                        continue  # finished a while ago
                    n = logf.stem.split("-")[-1]
                    icon = "✅" if role == "final" else "📟"
                    entry = f"{icon} суб-агент {n} · {role}⟩ {text}"
                    if len(entry) > self.TAIL_LINE_CAP:
                        entry = entry[: self.TAIL_LINE_CAP - 1] + "…"
                    out.append(entry)
            return out[:3]
        except Exception:
            return []

    async def refresh_tail_io(
        self,
        lines: list,
        *,
        active: bool,
        throttle_ok: bool,
        message_id,
        edit_fn,
        send_fn,
        logger=None,
    ):
        """Poll transcripts; edit/send the progress bubble if the tail changed.

        Runs from the queue-idle path of ``send_progress_messages``: while a
        delegated child is streaming its transcript, the bubble refreshes
        with the child's newest line even though no parent tool event
        arrived. Filesystem poll ~1s; ``throttle_ok`` carries the caller's
        flood-control throttle.

        Returns ``None`` when nothing was committed, else a tuple
        ``(message_id, can_edit)`` — the possibly-new progress message id and
        whether editing should stay enabled. On a committed refresh the
        caller must reset its edit-throttle timestamp.
        """
        try:
            if not (self.replace_mode and active):
                return None
            now = time.monotonic()
            if now - self._subagent_poll_ts < self.POLL_INTERVAL:
                return None
            self._subagent_poll_ts = now
            new_tail = self.collect_subagent_tail()
            sig = "\n".join(new_tail)
            if sig == self._subagent_sig or not throttle_ok:
                return None
            self._subagent_sig = sig
            self.subagent_tail = new_tail
            can_edit = True
            if message_id is not None:
                r = await edit_fn(message_id, self.render(lines))
                if not r.success and not getattr(r, "retryable", False):
                    can_edit = False
            elif new_tail:
                r = await send_fn(self.render(lines))
                if r.success and r.message_id:
                    message_id = r.message_id
            return message_id, can_edit
        except Exception as err:
            if logger is not None:
                logger.debug("subagent transcript tail refresh failed: %s", err)
            return None
