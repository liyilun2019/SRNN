import json
import numpy as np
import pickle
import pandas as pd

def get_subset():
	subset={}
	cnt=0;
	pdlist = pd.read_csv("../../data-beijing/beijing_roadSubset")
	for text in pdlist:
		subset[text.strip()]=cnt;
		cnt+=1
	print(subset);
	return subset

def get_W():
	subset = get_subset();
	with open('road_subset.pk','wb') as f:
		pickle.dump(subset,f);
		
	print(f"subset.size={len(subset)}");
	W=np.identity(len(subset))
	print(f"W.shape={W.shape}")
	with open('../../data-beijing/road_net.json','r') as f:
		new_dict = json.loads(f.read());
	for key in new_dict:
		if(key in subset):
			listO=new_dict[key]["o"];
			listD=new_dict[key]["d"];
			for item in listO:
				if(item in subset):
					o=subset[item]
					d=subset[key]
					W[o][d]=1
					W[d][o]=1
			for item in listD:
				if(item in subset):
					o=subset[key]
					d=subset[item]
					W[o][d]=1
					W[d][o]=1
	return W;
	# print(new_dict);


W=get_W();
print(W);
cnt=0;
for i in range(len(W)):
	for j in range(len(W[0])):
		if(W[i][j]!=0):
			cnt+=1
print(f"cnt={cnt}")
with open('W.pk', 'wb') as f:
	pickle.dump(W, f)
