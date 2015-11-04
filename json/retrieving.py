import cPickle as pkl
from scipy.sparse import vstack

def conc(matrix1, matrix2):
    new_data = numpy.concatenate((matrix1.data, matrix2.data))
    new_indices = numpy.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = numpy.concatenate((matrix1.indptr, new_ind_ptr))
    return csr_matrix((new_data, new_indices, new_ind_ptr))

path = 'real'

sp = []
for i in range(1,9):
	with open(path+'/train.pkl'+str(i*10000), 'rb') as f:
		print path+'/train.pkl'+str(i*10000)
		sp.append(pkl.load(f))

with open(path+'/train.pkl82783', 'rb') as f:
	sp.append(pkl.load(f))

sv = reduce(conc, sp)
del sp

with open(path+'/coco_align.train.feat.nd', 'wb') as f:
	pkl.dump(sv,f,protocol=pkl.HIGHEST_PROTOCOL)

