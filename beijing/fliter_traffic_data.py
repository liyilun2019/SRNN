import os
import pickle
import pandas as pd
import numpy as np

def read_pickle(path):
	with open(path,'rb') as f:
		data = pickle.load(f)
	return data

def read_one_file(path):
	with open(path,'r') as f:
		data = f.read().split()
	return data[0:2784]
	

subset = read_pickle('road_subset.pk');
newdic={}
filedir = '../../data-beijing/TrafficData';
for f in os.listdir(filedir):
	if(f in subset):
		ind = subset[f]
		newdic[ind]=read_one_file(filedir+'/'+f)
		print(f"{f}:{ind} with length = {len(newdic[ind])}")
# newdic=sorted(newdic.items(), key = lambda asd:asd[1], reverse = False)
df = pd.DataFrame(newdic)
df = df.T
df = df.sort_index(axis=0,ascending=True)
df = df.T
print(df.info())
df.to_csv('traffic_data0.csv')
df.to_csv('traffic_data.csv',index=None,header=None)
