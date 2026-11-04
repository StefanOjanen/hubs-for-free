# How the converted arms fail the retrieval task (round 35 and its development population): per arm, the answers
# that are empty (whitespace only), that carry any digit, that carry a five-digit number, that take the sentence
# form the fused development model produced ("The secret code for the <colour> door is ..."), that are degenerate
# (a phrase of three or more words repeated three times or more within the 24 generated tokens), and the commonest
# digit string among the misses. Reads alignment_study/retrieval_round/*.json and retrieval_round_dev/*.json and
# writes alignment_study/retrieval_round_answers.json; every number in the texts about the answers comes from it.
#   python alignment_study/retrieval_answers.py
import glob
import json
import os
import re
from collections import Counter

OUT = "alignment_study/retrieval_round_answers.json"
FILES = sorted(glob.glob("alignment_study/retrieval_round/*.json")) + sorted(glob.glob("alignment_study/retrieval_round_dev/*.json"))


def degenerate(a):
    words = a.split()
    for n in (3, 4, 5):
        grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
        if grams and grams.most_common(1)[0][1] >= 3:
            return True
    return False


def digits_of(a):
    tail = a.split(" is ")[-1] if " is " in a else a
    m = re.search(r"\d+", tail)
    return m.group(0) if m else None


def analyse(arm):
    inst = arm["needle"]["instances"]; ans = [i["answer"] for i in inst]; miss = [i["answer"] for i in inst if not i["hit"]]
    wrong = Counter(d for d in (digits_of(a) for a in miss) if d)
    return {"instances": len(inst), "hits": arm["needle"]["hits"], "empty": sum(not a.strip() for a in ans),
            "any_digit": sum(bool(re.search(r"\d", a)) for a in ans), "five_digit_number": sum(bool(re.search(r"\b\d{5}\b", a)) for a in ans),
            "sentence_form": sum(a.strip().startswith("The secret code") for a in ans), "degenerate": sum(degenerate(a) for a in ans),
            "misses_with_digits": sum(bool(re.search(r"\d", a)) for a in miss), "commonest_wrong_digits": wrong.most_common(3)}


if __name__ == "__main__":
    res = {}
    for fp in FILES:
        r = json.load(open(fp)); key = ("dev/" if "retrieval_round_dev" in fp else "") + os.path.basename(fp)[:-5]
        res[key] = {"model": r["model"], "registered": r.get("registered"), "arms": {k: analyse(v) for k, v in r["arms"].items()}}
    json.dump(res, open(OUT, "w"), indent=1)
    for key, r in res.items():
        print(key)
        for k, a in r["arms"].items():
            print(f"  {k:18s} hits {a['hits']:2d} of {a['instances']}; empty {a['empty']:2d}, any digit {a['any_digit']:2d}, five-digit {a['five_digit_number']:2d}, "
                  f"sentence form {a['sentence_form']:2d}, degenerate {a['degenerate']:2d}; commonest wrong digits {a['commonest_wrong_digits']}")
    print(f"RETRIEVAL_ANSWERS_DONE {OUT}")
