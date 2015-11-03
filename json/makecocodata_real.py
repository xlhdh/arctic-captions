import json
import pandas as pd
import numpy as np
import cPickle

sz = 512
path = str(sz)
path = "real"

# Load captions from MS
ja = json.loads(open('captions_val2014.json','r').read())['annotations']
jab = {j['image_id']:j['caption'] for j in ja}
captions = [j['caption'] for j in ja]


# Load image splits 
trainimages = open('../splits/coco_train.txt','r').read().splitlines()
valimages = open('../splits/coco_val.txt','r').read().splitlines()
testimages = open('../splits/coco_test.txt','r').read().splitlines()

# Making toy test data 
#trainimages = valimages[sz/2:sz]
#valimages = valimages[sz/4:sz/2]
#testimages = valimages[:sz/4]

# Make caps
cap_val = []
cap_train = []
cap_test = []
sp_train, sp_test, sp_val = [], [], []
captions = []

from scipy.io import loadmat
import scipy, numpy
## train.pkl: train
for idx, im in enumerate(trainimages):
	try: 
		data = loadmat('../coco_cnn4/'+im+'.mat')
		sp_train.append(data['o24'][0])
		cap_train.append((jab[int(im[19:25])], idx))
	except Exception:
		print "IMAGE NOT FOUND:",im
		pass
feat_train = scipy.sparse.csr_matrix(numpy.asarray(sp_train))
with open(path+'/coco_align.train.pkl', 'wb') as f:
    cPickle.dump(cap_train, f)
    cPickle.dump(feat_train, f)

## dev.pkl: val
for idx, im in enumerate(valimages):
	try: 
		data = loadmat('../coco_cnn4/'+im+'.mat')
		sp_val.append(data['o24'][0])
		cap_val.append((jab[int(im[19:25])], idx))
	except Exception:
		print "IMAGE NOT FOUND:",im
		pass
feat_val = scipy.sparse.csr_matrix(numpy.asarray(sp_val))
with open(path+'/coco_align.dev.pkl', 'wb') as f:
    cPickle.dump(cap_val, f)
    cPickle.dump(feat_val, f)

## test.pkl: test
for idx, im in enumerate(testimages):
	try: 
		data = loadmat('../coco_cnn4/'+im+'.mat')
		sp_test.append(data['o24'][0])
		cap_test.append((jab[int(im[19:25])], idx))
	except Exception:
		print "IMAGE NOT FOUND:",im
		pass
feat_test = scipy.sparse.csr_matrix(numpy.asarray(sp_test))
with open(path+'/coco_align.test.pkl', 'wb') as f:
    cPickle.dump(cap_test, f)
    cPickle.dump(feat_test, f)


# Making small dict for test 
# captions = [i[0] for i in cap_train]+[i[0] for i in cap_test]+[i[0] for i in cap_val]
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

#import shutil
#shutil.copy2(path+'/coco_align.dev.pkl', path+'/coco_align.train.pkl')
#shutil.copy2(path+'/coco_align.dev.pkl', path+'/coco_align.test.pkl')

