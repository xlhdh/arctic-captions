import cPickle as pkl
from numpy import load as lo
import numpy
from scipy.sparse import csr_matrix
from scipy.io import savemat


def conc(matrix1, matrix2):
	print "x"
	new_data = numpy.concatenate((matrix1.data, matrix2.data))
	new_indices = numpy.concatenate((matrix1.indices, matrix2.indices))
	new_ind_ptr = matrix2.indptr + len(matrix1.data)
	new_ind_ptr = new_ind_ptr[1:]
	new_ind_ptr = numpy.concatenate((matrix1.indptr, new_ind_ptr))
	return csr_matrix((new_data, new_indices, new_ind_ptr))

def ret80000():
	path = '/media/haboric/Ubuntu Data/yizhou/arctic-captions/json/real'

	sp = []
	for i in range(1,9):
		with open(path+'/train'+str(i*10000)+'.nd', 'rb') as f:
			sp.append(csr_matrix((lo(f),lo(f),lo(f))))

	#with open(path+'/train82783.nd', 'rb') as f:
	#	sp.append(csr_matrix((lo(f),lo(f),lo(f))))

	print 'before reduce'

	sv = reduce(conc, sp)
	del sp
	return sv

'''

print sv

a={}
a['o24']=sv

with open(path+'/coco_align.train.mat', 'wb') as f:
	savemat('temp',a)

with open(path+'/coco_align.train.feat.nd', 'wb') as f:
	pkl.dump(sv.indices,f,protocol=pkl.HIGHEST_PROTOCOL)
	pkl.dump(sv.indptr,f,protocol=pkl.HIGHEST_PROTOCOL)
with open(path+'/coco_align.train.feat-data.nd', 'wb') as f:
	pkl.dump(sv.data,f,protocol=pkl.HIGHEST_PROTOCOL)

,protocol=p.HIGHEST_PROTOCOL
'''