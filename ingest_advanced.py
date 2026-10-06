"""Build the indexes for the Advanced RAG use cases.

    python ingest_advanced.py            # all three
    python ingest_advanced.py gep imac   # selected ones
"""
import sys
import time

from advanced_rag import gep, gpo, imac


def main(names):
    for name in names:
        t = time.time()
        if name == "gpo":
            idx = gpo.build()
            idx.save("gpo")
        elif name == "gep":
            idx = gep.build()
            gep.GEP.save(idx)
        elif name == "imac":
            idx = imac.build()
            idx.save("imac")
        else:
            raise SystemExit(f"unknown use case '{name}' (choose from gpo, gep, imac)")
        print(f"{name}: {len(idx.chunks)} chunks indexed in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:] or ["gpo", "gep", "imac"])
