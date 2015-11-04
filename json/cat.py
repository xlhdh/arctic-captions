import json
#import cPickle

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

def maketrain():
	sp = []
	for idx, im in enumerate(trainimages):
		data = loadmat('../coco_cnn4/'+im)
		sp.append(csr_matrix(numpy.asarray(data['o24'])))
		if (idx % 10000) == 9999:
			print idx
			spv = vstack(sp,format='csr')
			with open(path+'/train'+str(idx+1)+'.nd', 'wb') as f:
				spv.data.dump(f)
				spv.indices.dump(f)
				spv.indptr.dump(f)
			sp = []
	
	spv = vstack(sp,format='csr')
	with open(path+'/train'+str(idx+1)+'.nd', 'wb') as f:
		spv.data.dump(f)
		spv.indices.dump(f)
		spv.indptr.dump(f)

	#COCO_train2014_000000286899.jpg
	return 0

if __name__ == "__main__":
	print "end reading"
	maketrain()
