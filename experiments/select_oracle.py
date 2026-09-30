"""
Pick, per trained model directory, the 'oracle' checkpoint: most random-heading
successes on its own training layout (out/checkpoints_*.json), ties broken by
deterministic success and then by the later episode. This is selected on the
evaluation layout, so results with it are an optimistic upper bound.
Writes out/oracle_checkpoints.json.
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
sel = {}
for f in sorted(glob.glob(os.path.join(HERE, "out", "checkpoints_*.json"))):
    r = json.load(open(f))
    for k, v in r["models"].items():
        kk = k.replace("\\", "/")
        grp = re.search(r"((?:env_apf_urban_|td3dyn_|td3_|sacdyn_|sac_)?seed\d+|models_original)", kk).group(1)
        m = re.search(r"ep(\d+)", kk)
        ep = int(m.group(1)) if m else -1          # arc_best ranks below equal-scoring epochs
        score = (v["random_heading_success"], v["deterministic"]["outcome"] == "success", ep)
        if grp not in sel or score > sel[grp][0]:
            sel[grp] = (score, kk)
for g, (s, k) in sorted(sel.items()):
    print(f"{g:24s} rand={s[0]}/10 det={'ok' if s[1] else 'fail'} -> {k}")
json.dump({g: k for g, (s, k) in sel.items()}, open(os.path.join(HERE, "out", "oracle_checkpoints.json"), "w"), indent=1)
