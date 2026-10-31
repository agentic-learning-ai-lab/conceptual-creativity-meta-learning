"""Print the Appendix C per-prompt baseline table (LaTeX rows) from the
output of experiments/analysis/baselines_dinov2.py."""
import json
import sys

COLS = [("CFG=1", "CFG $s{=}1$"), ("CFG=0.5", "CFG $s{=}0.5$"),
        ("Neg(strong/typical)", "Neg.\\ prompt"), ("ProCreate γ=250", "PC $\\gamma{=}250$"),
        ("ProCreate γ=500", "PC $\\gamma{=}500$"), ("ProCreate γ=1000", "PC $\\gamma{=}1000$")]

d = json.load(open(sys.argv[1]))
# Short names and grouping match the per-prompt reachability table in the paper.
SHORT = {"an alien botanical garden": "alien botanical garden",
         "a fortune teller's tent at twilight": "fortune teller's tent at twilight",
         "macro photograph of bioluminescent jellyfish in the deep ocean": "bioluminescent jellyfish",
         "stained-glass cathedral interior at night": "stained-glass cathedral interior"}


def short(p):
    return SHORT.get(p, p.removeprefix("a photo of an ").removeprefix("a photo of a "))


rows = {short(p): p for p in d["CFG=1"]["per_prompt"]}
imagenet = sorted(k for k, p in rows.items() if p.startswith("a photo of"))
open_ended = sorted(k for k, p in rows.items() if not p.startswith("a photo of"))
n = len(COLS) + 1
print("Prompt & " + " & ".join(c[1] for c in COLS) + " \\\\")
for title, group in [("ImageNet class prompts", imagenet), ("Open-ended prompts", open_ended)]:
    print("\\midrule")
    print(f"\\multicolumn{{{n}}}{{l}}{{\\textit{{{title}}}}} \\\\")
    for name in group:
        cells = []
        for key, _ in COLS:
            v = d[key]["per_prompt"].get(rows[name])
            cells.append("--" if v is None else f"{v['nn_dist_mean']:.2f}")
        print(f"{name} & " + " & ".join(cells) + " \\\\")
