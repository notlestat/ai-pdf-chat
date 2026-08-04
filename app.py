"""AI PDF Chat — upload a PDF, ask questions, get cited answers."""

from __future__ import annotations

import streamlit as st

from pdf_chat.chat import Answer, Conversation
from pdf_chat.client import MODEL, MissingAPIKey, get_client
from pdf_chat.documents import Document, PDFError, fingerprint, upload

st.set_page_config(page_title="AI PDF Chat", page_icon="📄", layout="centered")


@st.cache_resource(show_spinner=False)
def _client():
    return get_client()


def _reset_conversation() -> None:
    st.session_state.pop("conversation", None)
    st.session_state.pop("transcript", None)


def _load_document(uploaded) -> Document | None:
    """Upload the PDF once and reuse it across reruns.

    Streamlit re-executes this script top to bottom on every interaction, so
    without the fingerprint check the same PDF would be re-uploaded on every
    single question.
    """
    data = uploaded.getvalue()
    key = fingerprint(data)

    if st.session_state.get("doc_key") == key:
        return st.session_state["document"]

    with st.spinner(f"Reading {uploaded.name}…"):
        try:
            document = upload(_client(), data, uploaded.name)
        except PDFError as exc:
            st.error(str(exc))
            return None

    st.session_state["doc_key"] = key
    st.session_state["document"] = document
    _reset_conversation()
    return document


def _render_sources(answer: Answer) -> None:
    if not answer.citations:
        return
    pages = ", ".join(dict.fromkeys(c.label for c in answer.citations))
    with st.expander(f"Sources — {pages}"):
        for cite in answer.citations:
            st.markdown(f"**{cite.label}**")
            st.caption(f"“{cite.text}”")


def _render_reasoning(answer: Answer) -> None:
    if answer.reasoning:
        with st.expander("Model reasoning"):
            st.caption(answer.reasoning)


# --- Sidebar ----------------------------------------------------------------

with st.sidebar:
    st.subheader("Settings")
    effort = st.select_slider(
        "Reasoning effort",
        options=["low", "medium", "high"],
        value="high",
        help="Higher effort means more thorough reading and more tokens. "
        "Use high for contracts; low is fine for simple lookups.",
    )
    st.caption(f"Model: `{MODEL}`")

    conversation: Conversation | None = st.session_state.get("conversation")
    if conversation:
        conversation.effort = effort

    st.divider()
    st.subheader("Cost")

    if conversation and conversation.answers:
        last = conversation.answers[-1].usage
        st.metric("Session total", f"${conversation.total_cost:.3f}")
        if last:
            st.metric("Last question", f"${last.cost:.3f}")
            st.caption(
                f"in {last.input_tokens:,} · out {last.output_tokens:,}\n\n"
                f"cache read {last.cache_read_tokens:,} · "
                f"write {last.cache_write_tokens:,}"
            )
            # The document is the expensive part. Once it's cached, follow-ups
            # cost a fraction of the first question — this is the whole reason
            # the app sends the full PDF instead of chunking it.
            if last.cache_hit:
                st.success("Document served from cache", icon="⚡")
            elif len(conversation.answers) == 1:
                st.info("Document cached — follow-ups will be cheaper", icon="💾")
    else:
        st.caption("No questions asked yet.")

# --- Main -------------------------------------------------------------------

st.title("📄 AI PDF Chat")
st.caption("Ask questions about a PDF. Every answer cites the page it came from.")

try:
    _client()
except MissingAPIKey as exc:
    st.error(str(exc))
    st.stop()

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded is None:
    st.info("Upload a contract, handbook, manual, or paper to get started.")
    st.stop()

document = _load_document(uploaded)
if document is None:
    st.stop()

st.success(f"**{document.filename}** — {document.pages} pages", icon="✅")
if document.is_large:
    st.warning(
        f"This is a {document.pages}-page document. The first question reads the "
        "whole thing and will cost more than usual; follow-ups are cached and much "
        "cheaper. Watch the cost panel in the sidebar.",
        icon="⚠️",
    )

if "conversation" not in st.session_state:
    st.session_state["conversation"] = Conversation(_client(), document, effort)
    st.session_state["transcript"] = []

conversation = st.session_state["conversation"]

for entry in st.session_state["transcript"]:
    with st.chat_message(entry["role"]):
        st.markdown(entry["text"])
        if entry["role"] == "assistant" and entry.get("answer"):
            _render_sources(entry["answer"])
            _render_reasoning(entry["answer"])

question = st.chat_input("Ask about this document…")

if question:
    st.session_state["transcript"].append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            st.write_stream(conversation.ask(question))
        except Exception as exc:  # surfaced to the user rather than a stack trace
            st.error(f"Request failed: {exc}")
            st.stop()

        answer = conversation.answers[-1]
        _render_sources(answer)
        _render_reasoning(answer)

    st.session_state["transcript"].append(
        {"role": "assistant", "text": answer.text, "answer": answer}
    )
    st.rerun()
