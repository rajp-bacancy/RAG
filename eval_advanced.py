"""Check the three use cases against the questions from the challenge brief.

For each question we verify (1) retrieval: the context handed to the LLM contains the expected fact and
(2) with --answer: the generated answer contains it too. Run `python ingest_advanced.py` first.

    python eval_advanced.py [--answer] [gpo|gep|imac]
"""
import sys
import time

from advanced_rag.gep import GEP
from advanced_rag.gpo import GPO
from advanced_rag.imac import IMac

# (use case, question, [substrings that must all appear (case-insensitive)], optional section the top hits must come from)
CASES = [
    ("gpo", "How many depository libraries did GPO staff visit in FY 2018?", ["166"], None),
    ("gpo", "What new digital system replaced FDsys, and when was it fully deployed?", ["govinfo", "december 2018"], None),
    ("gpo", "How much did GPO procure in printing and information products in FY 2018 on behalf of federal agencies?", ["$375 million"], None),
    ("gep", "What is the growth outlook for South Asia in this report?", ["south asia"], "South Asia"),
    ("gep", "What does Chapter 3 say about sovereign debt in developing economies?", ["debt"], "Chapter 3"),
    ("gep", "According to the executive summary, what are the main risks to the global outlook?", ["risks"], "Executive Summary"),
    ("imac", "How much less energy does iMac use compared to the ENERGY STAR requirement?", ["58 percent"], None),
    ("imac", "What is the total carbon footprint of the iMac, and how much of that comes from product use versus manufacturing?", ["346", "45%", "50%"], None),
    ("imac", "What material is the iMac stand made from, and what percentage of it is recycled?", ["100 percent recycled aluminum"], None),
]
PIPES = {"gpo": GPO, "gep": GEP, "imac": IMac}


def main():
    want_answer = "--answer" in sys.argv
    only = [a for a in sys.argv[1:] if a in PIPES]
    loaded = {}
    failures = 0
    for uc, q, must, section in CASES:
        if only and uc not in only:
            continue
        pipe = loaded.setdefault(uc, PIPES[uc].load())
        if want_answer:
            answer, hits = pipe.answer(q)
            time.sleep(1)
        else:
            answer, hits = "", pipe.retrieve(q)
        context = " ".join(h["text"] for h in hits).lower()
        ok_ret = all(m.lower() in context for m in must)
        ok_sec = section is None or any(section.lower() in h["label"].lower() for h in hits[:3])
        ok_ans = (not want_answer) or all(m.lower().replace("percent", "") in answer.lower().replace("percent", "").replace("%", " ") or m.lower() in answer.lower() for m in must)
        status = "PASS" if (ok_ret and ok_sec and ok_ans) else "FAIL"
        failures += status == "FAIL"
        print(f"[{status}] ({uc}) {q}\n        retrieval={'ok' if ok_ret else 'MISSING'} section={'ok' if ok_sec else 'WRONG'}"
              + (f" answer={'ok' if ok_ans else 'MISSING'}" if want_answer else ""))
        print("        top hits:", " | ".join(f"p.{h['page_start']} {h['label'][-45:]}" for h in hits[:3]))
        if want_answer:
            print("        answer:", answer.replace("\n", " ")[:400])
    print(f"\n{failures} failing")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
