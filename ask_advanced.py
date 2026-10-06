"""Ask a question against one of the Advanced RAG use cases.

    python ask_advanced.py gep "What is the growth outlook for South Asia?"
    python ask_advanced.py gep "..." --retrieval-only    # just show what was retrieved
"""
import sys

from advanced_rag.gep import GEP
from advanced_rag.gpo import GPO
from advanced_rag.imac import IMac

PIPELINES = {"gpo": GPO, "gep": GEP, "imac": IMac}


def show(hits):
    for n, h in enumerate(hits, 1):
        tag = "neighbour" if h.get("neighbour") else ("attached" if h.get("attached_to") is not None else f"{h['score']:.4f}")
        print(f"  [{n}] {h['kind']:8} p.{h['page_start']}-{h['page_end']}  {h['label'][-80:]}  ({tag})")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2 or args[0] not in PIPELINES:
        raise SystemExit(__doc__)
    pipe = PIPELINES[args[0]].load()
    if "--retrieval-only" in sys.argv:
        hits = pipe.retrieve(args[1])
        print(hits[0].get("route", "") if hits else "")
        show(hits)
        return
    answer, hits = pipe.answer(args[1])
    print(answer, "\n\nSources:")
    show(hits)


if __name__ == "__main__":
    main()
