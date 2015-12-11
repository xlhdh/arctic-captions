import partition

image_prefix = 'COCO_train2014_'
file_path = '../../data/captions_train2014.json'
splits = '../../data/coco_train.txt'
output_name = 'indoor_train.txt'
wordSet = ['room', 'sofa', 'bed', 'table', 'chair', 'television']
#partition.filterDataSet(wordSet,file_path,output_name,splits,image_prefix)


image_prefix = 'COCO_val2014_'
file_path = '../../data/captions_val2014.json'
splits = '../../data/coco_test.txt'
output_name = 'indoor_test1.txt'
wordSet = ['room', 'sofa', 'bed', 'table', 'chair', 'television']
partition.filterDataSet(wordSet,file_path,output_name,splits,image_prefix)


image_prefix = 'COCO_val2014_'
file_path = '../../data/captions_val2014.json'
splits = '../../data/coco_val.txt'
output_name = 'indoor_val.txt'
wordSet = ['room', 'sofa', 'bed', 'table', 'chair', 'television']
#partition.filterDataSet(wordSet,file_path,output_name,splits,image_prefix)