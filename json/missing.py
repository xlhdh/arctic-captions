trainimages = open('../splits/coco_train.txt','r').read().splitlines()
valimages = open('../splits/coco_val.txt','r').read().splitlines()
testimages = open('../splits/coco_test.txt','r').read().splitlines()

import urllib

for filename in (trainimages+valimages+testimages):
	try:
	    with open('../coco_cnn4/'+filename+'.mat') as f:
	        pass
	except IOError as e:
		print filename
		#print filename, filename[21:27], int(filename[21:27]), str(int(filename[21:27]))
		#urllib.urlretrieve (('http://mscoco.org/images/'+str(int(filename[21:27]))), 'missfiles/'+filename)
		#print ('http://mscoco.org/images/'+str(int(filename[21:27])))