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
captions = [j['caption'] for j in ja]+[j['caption'] for j in jt]

# Load image splits 
trainimages = open('../splits/coco_train.txt','r').read().splitlines()
valimages = open('../splits/coco_val.txt','r').read().splitlines()
testimages = open('../splits/coco_test.txt','r').read().splitlines()

# Make caps
cap_val, cap_train, cap_test = [], [], []
sp_train, sp_test, sp_val = [], [], []

def makedict():
	# Making small dict for test 
	### Making dictionary 
	caps = []
	# TODO do lower case maybe? 
	for c in captions:
		caps.extend(c.split())
	print 1
	dictionary = {}
	while len(caps)>0:
		word  = caps.pop()
		dictionary[word] = dictionary.setdefault(word,0)+1
	#dictionary = {x:caps.count(x) for x in caps}
	print 2
	l = sorted(dictionary, key=lambda x:dictionary[x])
	print 3

	for idx, itm in enumerate(l):
		dictionary[itm]=idx+2
	print 4
	with open(path+'/dictionary.pkl', 'wb') as f:
	    cPickle.dump(dictionary, f)
	return 0
	### End making dictionary 


if __name__ == "__main__":
	print "end reading"
	makedict()
