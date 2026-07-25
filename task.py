"""Word-reversal task definition.

Shared by the dataset builder, the SFT trainer, the GRPO trainer, and the
evaluator so the prompt, the synthetic reasoning trace, and the answer
parsing can never drift apart.
"""

import re

PROMPT_TEMPLATE = (
    'Reverse the word "{word}". Reason in <think> XML tags, '
    "and put your answer in <answer> tags."
)

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def make_prompt(word: str) -> str:
    return PROMPT_TEMPLATE.format(word=word)


def make_trace(word: str) -> str:
    """Synthesize the gold reasoning trace for SFT.

    Format:

        <think>
        "pluto" is spelled "p", "l", "u", "t", "o". That's:

        1. p
        2. l
        3. u
        4. t
        5. o
        So reversed is 5 -> o, 4 -> t, 3 -> u, 2 -> l, 1 -> p, or "o", "t",
        "u", "l", "p". Which is "otulp." Answer is "otulp".
        </think>
        <answer>
        otulp
        </answer>
    """
    letters = list(word)
    rev = word[::-1]
    n = len(letters)
    spelled = ", ".join(f'"{c}"' for c in letters)
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(letters))
    rev_steps = ", ".join(f"{i} -> {letters[i - 1]}" for i in range(n, 0, -1))
    rev_letters = ", ".join(f'"{c}"' for c in rev)
    return (
        "<think>\n"
        f'"{word}" is spelled {spelled}. That\'s:\n'
        "\n"
        f"{numbered}\n"
        f'So reversed is {rev_steps}, or {rev_letters}. '
        f'Which is "{rev}." Answer is "{rev}".\n'
        "</think>\n"
        "<answer>\n"
        f"{rev}\n"
        "</answer>"
    )


def parse_answer(text: str):
    """Extract the contents of the first <answer> tag, or None if absent."""
    m = ANSWER_RE.search(text)
    if m is None:
        return None
    return m.group(1).strip()


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
