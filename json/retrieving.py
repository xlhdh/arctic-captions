import cPickle as pkl
from scipy.sparse import vstack

path = 'real'

sp = []
for i in range(1,9):
	with open(path+'/train.pkl'+str(i*10000), 'wb') as f:
		    sp.append(pkl.load(f))

with open(path+'/train.pkl82783', 'wb') as f:
	    sp.append(pkl.load(f))

with open(path+'/cat.pkl', 'wb') as f:
	    cat = pkl.load(f)

with open(path+'/coco_align.train.pkl', 'wb') as f:
	    pkl.dump(cat,f)
	    pkl.dump(vstack(sp),f)


