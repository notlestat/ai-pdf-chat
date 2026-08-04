# 📄 AI PDF Chat

Upload a PDF, ask questions in plain English, get answers **with the page number they came from**.

Built for documents people are accountable for — contracts, employee handbooks, policies,
technical manuals, research papers — where a confidently wrong answer is worse than no answer.

```javascript
Upload PDF ──▶ Files API (uploaded once)
                    │
                    ▼
       document block { citations: on, cache: on }
                    │
                    ▼
             Claude Opus 5 ── streams the answer
                    │
                    ▼
        Answer + the exact passage and page it came from
```

---

## Why this reads the whole document instead of using a vector database

The default architecture for "chat with your PDF" is RAG: chop the document into chunks, embed
them, store them in a vector database, and retrieve the top few chunks for each question.

This app doesn't do that. It sends the entire PDF to the model on every conversation. That is a
deliberate trade, and here is the reasoning:

|                                      | Whole document (this app)              | Chunk + vector search                              |
| ------------------------------------ | -------------------------------------- | -------------------------------------------------- |
| "When does this contract expire?"    | Reads all 200 pages, finds it          | Works **if** the clause lands in the top-k results |
| "Summarise section 4"                | Reliable                               | Weak — chunking destroys document structure        |
| Facts spread across several sections | Reliable                               | Frequently misses                                  |
| Citations                            | Exact page numbers, built into the API | You build it yourself, chunk-level                 |
| Infrastructure                       | None                                   | Vector DB to run, re-index, and back up            |
| Cost per follow-up question          | \~10–20¢                               | \~2–3¢                                             |
| Cost to load a 200-page contract     | \~$2–3, once per session               | Pennies                                            |

**RAG is genuinely cheaper. Whole-document reading is genuinely more correct.**

For the intended user, correctness wins and it isn't close. A solicitor billing hundreds an hour
does not care about $3; they care that the expiry date is right and that they can click through
to page 47 and verify it. Retrieval failures in this setting are silent — you get a fluent,
plausible, wrong answer with no signal that the relevant clause was never retrieved.

**When I would switch to RAG:** a corpus of many documents rather than one, or consumer-scale
traffic where per-question cost dominates the economics. The chunking cost is worth paying
when you cannot fit the corpus in context; it is not worth paying when you can.

### Making it affordable anyway

Reading a whole document on every question would be wasteful, so the app relies on
[prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching). The document
block sits in the first message and never moves, so everything up to and including the PDF forms
a stable cached prefix. The first question pays to process the document; every follow-up reads it
back at roughly a tenth of the price.

The sidebar shows this happening live — token counts, cache hits, and running cost per question.

---

## What it does

- **Cited answers.** Every claim links to a page range, with the quoted passage shown.
- **Refuses to guess.** If the document doesn't say, it says so instead of inventing something.
- **Live cost tracking.** Per-question and session cost, plus cache hit/miss.
- **Adjustable reasoning effort.** Trade thoroughness against cost from the sidebar.
- **Visible reasoning.** Expand to see how the model worked the answer out.

---

## Quickstart

```bash
git clone https://github.com/notlestat/ai-pdf-chat.git
cd ai-pdf-chat

# uv (recommended)
uv venv
uv pip install -r requirements.txt

# or plain pip
python -m venv .venv && .venv/Scripts/activate   # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Add your API key:

```bash
cp .env.example .env        # Windows: copy .env.example .env
# then edit .env and paste your key from console.anthropic.com
```

Run it:

```bash
streamlit run app.py
```

---

## Project layout

```javascript
app.py                 Streamlit UI — upload, chat, sources, cost panel
pdf_chat/
├── client.py          API client + the TLS fix described below
├── documents.py       PDF validation and one-time upload
└── chat.py            Prompt assembly, caching, streaming, citation parsing
```

The Anthropic integration lives entirely outside `app.py`, so it can be exercised and tested
without starting Streamlit.

---

## Troubleshooting

### `CERTIFICATE_VERIFY_FAILED` on every request

If Python cannot reach *any* HTTPS endpoint while your browser and `curl` work fine, an
antivirus product is almost certainly intercepting TLS. I hit this building the app on a
machine running Norton.

The cause: these products install a local root CA and re-sign every certificate. Norton's CA
declares `Basic Constraints` without marking the extension **critical**, and OpenSSL 3.x rejects
that outright — so Python fails while Windows' own certificate stack, which is more lenient,
accepts it happily.

The fix is [`truststore`](https://pypi.org/project/truststore/), applied in `pdf_chat/client.py`
before any network call:

```python
import truststore
truststore.inject_into_ssl()
```

This delegates verification to the operating system instead of OpenSSL. Certificates are still
fully verified — just by a verifier that handles the malformed CA. Note this is **not** the same
as `verify=False`, which disables verification altogether and should never ship.

If `uv` itself fails to install packages for the same reason, use `uv pip install --system-certs`.

### Answers have no citations

Citations only attach to claims drawn from the document. A question the PDF doesn't cover
produces an uncited "the document doesn't specify" — which is the intended behaviour.

---

## What I'd change at scale

- **Many documents** — this is the point where chunking and a vector index start earning their
  keep, with whole-document reading kept for the final answer.
- **Scanned PDFs** — Claude reads page images, so these mostly work, but a text layer via OCR
  would make page-count validation and search more reliable.
- **Longer cache TTL** — the 5-minute cache refreshes as long as the user keeps asking. For
  sessions with long reading pauses, the 1-hour TTL would cost more to write but stop the
  document falling out of cache mid-session.
- **Model routing** — simple lookups don't need Opus. Classifying question difficulty and
  routing the easy ones to Sonnet would cut cost meaningfully.

---

## Built with

[Claude Opus 5](https://docs.claude.com/en/docs/about-claude/models) · Anthropic Files API ·
prompt caching · citations · [Streamlit](https://streamlit.io) · [pypdf](https://pypdf.readthedocs.io)
