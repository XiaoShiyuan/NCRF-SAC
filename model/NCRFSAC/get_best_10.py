"""
Original author: Xiao Shiyuan

search 10 checkpoints with highest word-level F1 score
"""

log = []
with open('results/main.log', 'r', encoding='UTF8') as f:
	for line in f.readlines():
		if line.startswith('Saving dict'):
			log += [line.strip()]

log.reverse()
d = {}
for line in log[:100]:   
	f1 = line.split(',')[1].strip()
	check = line.split(',')[0].strip().split()
	for i in check:
		if i[0].isdigit():
			e = i
			break
	score = float(f1.split()[2])
	d[i] = score
emmm = sorted(d.items(), key=lambda x:x[1], reverse=True)
print(emmm[:10])
