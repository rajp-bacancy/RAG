"""Streamlit UI for the Advanced RAG challenge: pick a document, ask questions, inspect the retrieved sources."""
import logging

import streamlit as st

from advanced_rag.gep import GEP
from advanced_rag.gpo import GPO
from advanced_rag.imac import IMac

DOCS = {
    "GPO 2018 Annual Report (narrative text)": (GPO, "How many depository libraries did GPO staff visit in FY 2018?"),
    "World Bank GEP June 2026 (long, structured)": (GEP, "What is the growth outlook for South Asia in this report?"),
    "Apple iMac Environmental Report (charts, tables, footnotes)": (IMac, "How much less energy does iMac use compared to the ENERGY STAR requirement?"),
}

st.set_page_config(page_title="Advanced RAG", page_icon="📚")
st.title("📚 Advanced RAG")
choice = st.sidebar.selectbox("Document", list(DOCS))
cls, example = DOCS[choice]


@st.cache_resource
def load(name: str):
    return DOCS[name][0].load()


try:
    pipe = load(choice)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.caption(f"Example: {example}")
question = st.chat_input("Ask a question about this document")
if question:
    st.chat_message("user").write(question)
    with st.spinner("Retrieving and answering..."):
        try:
            answer, hits = pipe.answer(question)
        except Exception:  # network / API-key problems; details stay in the server log, not the page
            logging.getLogger(__name__).exception("answer failed")
            st.error("Could not generate an answer. Check your GOOGLE_API_KEY and network, then try again.")
            st.stop()
    # "$" pairs would otherwise be rendered as LaTeX math by Streamlit
    st.chat_message("assistant").write(answer.replace("$", "\\$"))
    with st.expander(f"Retrieved context ({len(hits)} chunks)"):
        if hits and hits[0].get("route"):
            st.caption(f"Routing: {hits[0]['route']}")
        for h in hits:
            note = " - neighbour" if h.get("neighbour") else (" - attached footnote" if h.get("attached_to") is not None else "")
            st.markdown(f"**{h['kind']}** · p.{h['page_start']}-{h['page_end']} · {h['label']}{note}")
            st.text(h["text"][:900])
