"""Stand-ins for the language model."""

from __future__ import annotations

from collections.abc import Sequence


class ScriptedLLM:
    """Return queued strings, one per ``complete`` call."""

    def __init__(self, responses: Sequence[str]):
        self._responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages, *, temperature: float = 0.0, json_mode: bool = False) -> str:
        self.calls.append([dict(message) for message in messages])
        if not self._responses:
            raise AssertionError("ScriptedLLM has no responses left")
        return self._responses.pop(0)
