
import subprocess, itertools
for m in ["bert-base-uncased","bert-base-cased"]:
    subprocess.run(["python","src/train_model.py","--model_name",m])
