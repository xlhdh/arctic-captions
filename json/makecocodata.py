import json
import pandas as pd
import numpy as np
import cPickle

sz = 256
path = str(sz)

# Load captions from MS
ja = json.loads(open('captions_val2014.json','r').read())['annotations']
#captions = [j['caption'] for j in ja]


# Load image splits 
valimages = open('../splits/coco_val.txt','r').read().splitlines()
valimages = valimages[:sz]
valimageids = [int(i[19:25]) for i in valimages]

# Make caps
cap_val = []
captions = []
sp = []
for it in ja: 
	if it['image_id'] in valimageids:
		cap_val.append((it['caption'], valimageids.index(it['image_id'])))
		# TODO to be consisitent with CNN feats
		captions.append(it['caption'])
# End making caps

# Make CNN features 
from scipy.io import loadmat
import scipy, numpy
sp = []
for im in valimages:
	data = loadmat('../coco_cnn4/'+im)
	sp.append(data['o24'][0])
feat_val = scipy.sparse.csr_matrix(numpy.asarray(sp))
# End making CNN features 
with open(path+'/coco_align.dev.pkl', 'wb') as f:
    cPickle.dump(cap_val, f)
    cPickle.dump(feat_val, f)


### Making dictionary 
# from nltk import word_tokenize as wt
caps = []
# TODO do lower case maybe? 
for c in captions:
	caps.extend(c.split())
dictionary = {x:caps.count(x) for x in caps}
l = sorted(dictionary, key=lambda x:dictionary[x])

for idx, itm in enumerate(l):
	dictionary[itm]=idx+2

with open(path+'/dictionary.pkl', 'wb') as f:
    cPickle.dump(dictionary, f)
### End making dictionary 

import shutil
shutil.copy2(path+'/coco_align.dev.pkl', path+'/coco_align.train.pkl')
shutil.copy2(path+'/coco_align.dev.pkl', path+'/coco_align.test.pkl')

