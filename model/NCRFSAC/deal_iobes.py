file = "results/score/testb.preds.txt"

tmp = []
with open(file, "r", encoding="utf8") as f:
    for line in f.readlines():
        tmp.append(line.replace("E-", "I-").replace("S-", "B-"))

with open(file, "w", encoding="utf8") as f:
    f.writelines(tmp)
