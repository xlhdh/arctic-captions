import json
js = open('captions_val2014.json','r').read()
j1 = json.loads(js)

ja = j1['annotations']

l = [i['file_name'] for i in ji]

'''
l.sort()

for x in l:
    print x
'''


spli = open('../splits/coco_val.txt','r').read().splitlines()

