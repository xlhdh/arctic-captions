import cPickle as pkl
from scipy.sparse import vstack

path = 'real'

sp = []
for i in range(1,9):
	with open(path+'/train.pkl'+str(i*10000), 'wb') as f:
		sp.append(pkl.load(f))

with open(path+'/train.pkl82783', 'wb') as f:
	sp.append(pkl.load(f))

sp = vstack(sp,format='csr')

with open(path+'/coco_align.train.feat.nd', 'wb') as f:
	sp.data.dump(f)
	sp.indices.dump(f)
	sp.indptr.dump(f)

