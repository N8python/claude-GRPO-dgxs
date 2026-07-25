"""Build the reversal-task dataset (the data/ directory ships pre-built;
run this only if you want to regenerate it from scratch).

Takes the top 10,000 English words of length >= 5 by frequency (Norvig's
Google-corpus unigram counts, already sorted by count) and splits with fixed
seeds:

    eval  1,000   held-out evaluation words
    rl    1,000   prompts for GRPO
    sft     100   words whose gold traces form the SFT set

The SFT split is deliberately tiny: 100 traces teach the FORMAT; learning to
actually reverse words is the RL phase's job.

Outputs (under data/):
    eval_words.txt, rl_words.txt, sft_words.txt   one word per line
    sft_train.jsonl                               {"messages": [...]} chat rows
"""

import json
import random
import urllib.request
from pathlib import Path

from task import make_prompt, make_trace

WORDLIST_URL = "https://norvig.com/ngrams/count_1w.txt"
MIN_LEN = 5
TOP_N = 10_000
SPLIT_SEED = 42
RL_SEED = 0
RL_SIZE = 1_000

DATA = Path(__file__).parent / "data"


def get_words():
    cache = DATA / "count_1w.txt"
    if not cache.exists():
        print(f"downloading {WORDLIST_URL} ...")
        urllib.request.urlretrieve(WORDLIST_URL, cache)
    words = []
    with open(cache) as f:
        for line in f:
            word = line.split("\t")[0].strip()
            if len(word) >= MIN_LEN and word.isalpha() and word.isascii():
                words.append(word.lower())
            if len(words) == TOP_N:
                break
    assert len(words) == TOP_N, f"only found {len(words)} words"
    assert len(set(words)) == TOP_N, "duplicate words in source list"
    return words


def main():
    DATA.mkdir(exist_ok=True)
    words = get_words()
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(words)

    splits = {
        "eval": sorted(words[:1_000]),
        "sft": sorted(words[9_900:]),
    }
    # the RL pool: a fixed 1,000-word sample of the remaining 8,900
    pool = sorted(words[1_000:9_900])
    random.Random(RL_SEED).shuffle(pool)
    splits["rl"] = sorted(pool[:RL_SIZE])

    for name in ["eval", "rl", "sft"]:
        split_words = splits[name]
        out = DATA / f"{name}_words.txt"
        out.write_text("\n".join(split_words) + "\n")
        lens = [len(w) for w in split_words]
        print(f"{name}: {len(split_words)} words, len {min(lens)}-{max(lens)} "
              f"(mean {sum(lens)/len(lens):.1f}) -> {out}")

    sft_path = DATA / "sft_train.jsonl"
    with open(sft_path, "w") as f:
        for word in splits["sft"]:
            row = {
                "messages": [
                    {"role": "user", "content": make_prompt(word)},
                    {"role": "assistant", "content": make_trace(word)},
                ]
            }
            f.write(json.dumps(row) + "\n")
    print(f"sft traces -> {sft_path}")


if __name__ == "__main__":
    main()
