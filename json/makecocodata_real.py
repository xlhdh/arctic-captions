import json
import cPickle

path = "real"

# Load captions from MS
ja = json.loads(open('captions_val2014.json','r').read())['annotations']
jt = json.loads(open('captions_train2014.json','r').read())['annotations']
#jab = {j['image_id']:j['caption'] for j in ja}
#jab.update({j['image_id']:j['caption'] for j in jt})
print "json"
jab = {}
for k in (ja+jt):
	if k['image_id'] in jab:
		jab[k['image_id']].append(k['caption'])
	else:
		jab[k['image_id']] = [k['caption'],]
print "jabs"
#captions = [j['caption'] for j in ja]+[j['caption'] for j in jt]

# Load image splits 
trainimages = open('../splits/coco_train.txt','r').read().splitlines()
valimages = open('../splits/coco_val.txt','r').read().splitlines()
testimages = open('../splits/coco_test.txt','r').read().splitlines()

# Make caps
cap_val, cap_train, cap_test = [], [], []
sp_train, sp_test, sp_val = [], [], []
captions = []

from scipy.io import loadmat
import scipy, numpy
## train.pkl: train
def maketrain():
	for idx, im in enumerate(trainimages):
		if (idx % 100) == 0:
			print idx, im
		data = loadmat(('../coco_cnn4/'+im), appendmat=True)
		sp_train.append(data['o24'][0])
	feat_train = scipy.sparse.csr_matrix(numpy.asarray(sp_train))
	with open(path+'/coco_align.train.pkl', 'wb') as f:
	    cPickle.dump(cap_train, f)

	for idx, im in enumerate(trainimages):
		if (idx % 100) == 0:
			print idx, im
		for j in jab[int(im[21:27])]:
			cap_train.append((j, idx))	
	f = open(path+'/coco_align.train.pkl', 'wb')
	cPickle.dump(feat_train, f)
	return 0

## dev.pkl: val
def makeval():
	for idx, im in enumerate(valimages):
		print idx, im
		data = loadmat(('../coco_cnn4/'+im), appendmat=True)
		sp_val.append(data['o24'][0])
		for j in jab[int(im[19:25])]:
			cap_val.append((j, idx))
	feat_val = scipy.sparse.csr_matrix(numpy.asarray(sp_val))
	with open(path+'/coco_align.dev.pkl', 'wb') as f:
	    cPickle.dump(cap_val, f)
	    cPickle.dump(feat_val, f)
	return 0

## test.pkl: test
def maketest():
	for idx, im in enumerate(testimages):
		print idx, im
		data = loadmat(('../coco_cnn4/'+im), appendmat=True)
		sp_test.append(data['o24'][0])
		for j in jab[int(im[19:25])]:
			cap_test.append((j, idx))
	feat_test = scipy.sparse.csr_matrix(numpy.asarray(sp_test))
	with open(path+'/coco_align.test.pkl', 'wb') as f:
	    cPickle.dump(cap_test, f)
	    cPickle.dump(feat_test, f)
	return 0


def makedict():
	# Making small dict for test 
	### Making dictionary 
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
	return 0
	### End making dictionary 


if __name__ == "__main__":
	print "end reading"
	maketrain()
