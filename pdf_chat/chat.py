"""Conversation handling: prompt assembly, streaming, citations, cost."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

import anthropic

from pdf_chat.client import FILES_BETA, MODEL
from pdf_chat.documents import Document

Effort = Literal["low", "medium", "high"]

# The refusal instruction is the most important line here. This app is aimed at
# contracts and policies, where an invented answer is worse than no answer — a
# lawyer who gets a plausible wrong expiry date is worse off than one who gets
# "the document doesn't say".
SYSTEM_PROMPT = """You answer questions about a single PDF document the user has provided.

Ground every claim in the document. Quote it directly when the exact wording matters — \
in contracts and policies the precise phrasing usually is the answer.

If the document does not contain the answer, say so plainly and stop. Do not infer, \
estimate, or fall back on general knowledge about how documents like this usually read. \
"The document doesn't specify" is a correct and useful answer; a plausible guess is not.

If the document is ambiguous or says contradictory things in different places, say that \
and show both passages rather than silently picking one.

Answer at the length the question deserves. A question about a single date warrants a \
sentence, not an essay."""

# Claude Opus 5, USD per million tokens.
PRICE_INPUT = 5.00
PRICE_OUTPUT = 25.00
PRICE_CACHE_READ = 0.50  # ~0.1x input
PRICE_CACHE_WRITE = 6.25  # ~1.25x input, 5-minute TTL


@dataclass(frozen=True)
class Citation:
    """A passage the model quoted, and where it came from."""

    text: str
    start_page: int
    end_page: int

    @property
    def label(self) -> str:
        if self.start_page == self.end_page:
            return f"Page {self.start_page}"
        return f"Pages {self.start_page}–{self.end_page}"


@dataclass(frozen=True)
class Usage:
    """Token counts and cost for one exchange."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int

    @property
    def cost(self) -> float:
        return (
            self.input_tokens * PRICE_INPUT
            + self.output_tokens * PRICE_OUTPUT
            + self.cache_read_tokens * PRICE_CACHE_READ
            + self.cache_write_tokens * PRICE_CACHE_WRITE
        ) / 1_000_000

    @property
    def cache_hit(self) -> bool:
        """True when the document was served from cache instead of re-read."""
        return self.cache_read_tokens > 0


@dataclass
class Answer:
    """Everything produced by one question, available after the stream finishes."""

    text: str = ""
    citations: list[Citation] = field(default_factory=list)
    reasoning: str = ""
    usage: Usage | None = None


class Conversation:
    """A chat session about one document.

    The document block sits in the first user turn and never moves. That ordering
    is what makes prompt caching work: the cached prefix is everything up to and
    including the document, so each follow-up question re-reads it from cache at
    roughly a tenth of the price instead of paying to process the PDF again.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        document: Document,
        effort: Effort = "high",
    ) -> None:
        self.client = client
        self.document = document
        self.effort: Effort = effort
        self._messages: list[dict[str, Any]] = []
        self.answers: list[Answer] = []

    def _user_turn(self, question: str) -> dict[str, Any]:
        """Build a user message, attaching the document on the first turn only."""
        if self._messages:
            return {"role": "user", "content": question}

        return {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "file", "file_id": self.document.file_id},
                    "title": self.document.filename,
                    # Makes the model attribute each claim to a page range, which
                    # is what makes the answer checkable rather than just fluent.
                    "citations": {"enabled": True},
                    # The single cache breakpoint. Everything before it — system
                    # prompt plus the whole PDF — is cached from here on.
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": question},
            ],
        }

    def ask(self, question: str) -> Iterator[str]:
        """Ask a question, yielding the answer as it arrives.

        Streams rather than blocking: a large PDF can take a while to process, and
        a non-streaming request risks an HTTP timeout on top of the bad UX.
        """
        self._messages.append(self._user_turn(question))
        answer = Answer()

        with self.client.beta.messages.stream(
            model=MODEL,
            max_tokens=16_000,
            betas=[FILES_BETA],
            system=SYSTEM_PROMPT,
            # Opus 5 thinks by default. Asking for summarized display means the UI
            # can show progress; with the default "omitted" the thinking blocks
            # stream as empty text and the app just looks frozen.
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.effort},
            messages=self._messages,
        ) as stream:
            for chunk in stream.text_stream:
                answer.text += chunk
                yield chunk
            final = stream.get_final_message()

        # Echo the response back into history so the model keeps its own context,
        # including thinking blocks, which must survive unchanged.
        self._messages.append(
            {"role": "assistant", "content": _for_history(final.content)}
        )

        answer.citations = _citations(final.content)
        answer.reasoning = _reasoning(final.content)
        answer.usage = _usage(final.usage)
        self.answers.append(answer)

    @property
    def total_cost(self) -> float:
        return sum(a.usage.cost for a in self.answers if a.usage)


def _for_history(content: list[Any]) -> list[dict[str, Any]]:
    """Convert a response back into blocks the API will accept as input.

    Response and request shapes are not symmetric, and every mismatch here shows
    up on the *second* question rather than the first:

    1. Citations are output metadata, not input. They carry a `file_id` and
       positional `document_index` values that the API re-validates against the
       documents in the request, and they do not survive the round trip
       ("Invalid citation indices", "Extra inputs are not permitted"). We keep
       them for display and drop them from history — the answer text is the
       context the model actually needs.
    2. With citations on, the reply is split into cited and uncited segments and
       the trailing one is often empty. The API emits those but refuses them as
       input ("text content blocks must be non-empty").

    Everything else passes through untouched — thinking blocks in particular
    must keep their signatures.
    """
    blocks: list[dict[str, Any]] = []

    for block in content:
        data = (
            block.model_dump(exclude_none=True)
            if hasattr(block, "model_dump")
            else dict(block)
        )

        if data.get("type") == "text":
            if not (data.get("text") or "").strip():
                continue
            data.pop("citations", None)

        blocks.append(data)

    return blocks


def _citations(content: list[Any]) -> list[Citation]:
    """Pull page-level citations out of the response, keeping first-seen order."""
    found: list[Citation] = []
    seen: set[tuple[int, int, str]] = set()

    for block in content:
        for cite in getattr(block, "citations", None) or []:
            # PDFs yield page_location; other document types use different shapes.
            if getattr(cite, "type", None) != "page_location":
                continue
            start = cite.start_page_number
            # end_page_number is exclusive, so a single-page citation reports
            # start=7, end=8. Normalise to an inclusive range for display.
            end = max(start, cite.end_page_number - 1)
            text = (cite.cited_text or "").strip()

            key = (start, end, text)
            if key in seen:
                continue
            seen.add(key)
            found.append(Citation(text=text, start_page=start, end_page=end))

    return found


def _reasoning(content: list[Any]) -> str:
    parts = [
        block.thinking
        for block in content
        if getattr(block, "type", None) == "thinking" and getattr(block, "thinking", "")
    ]
    return "\n\n".join(parts).strip()


def _usage(usage: Any) -> Usage:
    return Usage(
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )
