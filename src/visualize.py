
import json, pathlib, matplotlib.pyplot as plt
recs=[json.loads(l) for l in pathlib.Path("data/processed/harmonised/physionet.jsonl").open()]
lengths=[len(r["tokens"]) for r in recs]
plt.bar(range(len(lengths)), lengths); plt.ylabel("Tokens"); plt.title("Length per record"); plt.tight_layout(); plt.savefig("slides/token_lengths.png")
