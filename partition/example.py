import partition

image_prefix = 'COCO_train2014_'
file_path = '../../data/captions_train2014.json'
splits = '../../data/coco_train.txt'
output_name = 'animals_train.txt'
wordSet = ['animal', 'cat', 'dog', 'horse', 'bird', 'monkey']
partition.filterDataSet(wordSet,file_path,output_name,splits,image_prefix)


image_prefix = 'COCO_val2014_'
file_path = '../../data/captions_val2014.json'
splits = '../../data/coco_test.txt'
output_name = 'animals_test.txt'
wordSet = ['animal', 'cat', 'dog', 'horse', 'bird', 'monkey']
partition.filterDataSet(wordSet,file_path,output_name,splits,image_prefix)


image_prefix = 'COCO_val2014_'
file_path = '../../data/captions_val2014.json'
splits = '../../data/coco_val.txt'
output_name = 'animals_val.txt'
wordSet = ['animal', 'cat', 'dog', 'horse', 'bird', 'monkey']
partition.filterDataSet(wordSet,file_path,output_name,splits,image_prefix)