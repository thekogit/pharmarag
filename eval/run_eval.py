import json, os, sys, time
from src.orchestrator import expand_node, retrieve_node, synth_node

REFUSAL = "do not contain sufficient information"

def run(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        q = json.loads(line)
        t0 = time.time()
        state = retrieve_node(expand_node({"question": q["question"], "expanded": [], "docs": [], "answer": ""}))
        answer = synth_node(state)["answer"]
        ranks = [i + 1 for i, d in enumerate(state["docs"])
                 if d["meta"].get("compound") == q["compound"] and f"[Section: {q['section']}]" in d["text"]]
        facts = [s.lower() in answer.lower() for s in q["must_contain"]]
        rows.append({
            "id": q["id"], "unanswerable": q["unanswerable"],
            "hit": 1 if ranks else 0, "rr": 1 / ranks[0] if ranks else 0.0,
            "fact_recall": sum(facts) / len(facts) if facts else 0.0,
            "refused": int(REFUSAL in answer), "latency_s": round(time.time() - t0, 1),
        })
    return rows

def summary(rows, config):
    ans = [r for r in rows if not r["unanswerable"]]
    un = [r for r in rows if r["unanswerable"]]
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else 0.0
    lat = sorted(r["latency_s"] for r in rows)
    return (f"| {config} | {mean([r['hit'] for r in ans])} | {mean([r['rr'] for r in ans])} | "
            f"{mean([r['fact_recall'] for r in ans])} | {sum(r['refused'] for r in un)}/{len(un)} | "
            f"{sum(r['refused'] for r in ans)} | {lat[len(lat) // 2]} |")

if __name__ == "__main__":
    config = f"{os.getenv('RETRIEVAL_MODE', 'hybrid')}, rerank={os.getenv('RERANK', '1')}"
    rows = run(sys.argv[1] if len(sys.argv) > 1 else "eval/questions.jsonl")
    json.dump(rows, open(f"eval/rows_{config.replace(', ', '_').replace('=', '')}.json", "w"), indent=1)
    print("| Config | Hit@5 | MRR | Fact recall | Correct refusals | False refusals | Median latency (s) |")
    print("|---|---|---|---|---|---|---|")
    print(summary(rows, config))
