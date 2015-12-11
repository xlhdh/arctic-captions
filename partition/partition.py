import json
import random
import pattern.en

def filterDataSet(wordSet, file_path, output_name, splits, image_prefix, noise = 0.1):
    pluralForm = []
    for x in wordSet:
        a = pattern.en.pluralize(x)
        pluralForm.append(a)

    wordSet.append(pluralForm)

    splitsSet = set()
    with open(splits,'r') as f:
        for line in f:
            splitsSet.add(line)

    ja = json.loads(open(file_path, 'r').read())    # ['annotations']
    annotations = ja['annotations']
    # print annotations[0]
    # print images[0]

    imageSet = set()
    noiseSet = set()

    random.seed()

    testSet = set(['cat','laptop'])
    for a in annotations:
        #if a['caption'] == 'A cat laying on top of a laptop computer.':
        #    print a['image_id']
        aSplit = a['caption'].split()
        if all(word in aSplit for word in testSet):
            print a['caption']

        imageName = image_prefix + str(a['image_id']).zfill(12) + '.jpg\n'
        if not(imageName in splitsSet):
            continue

        if any(word in aSplit for word in wordSet):
            print a['caption']
            imageSet.add(imageName)
        else:
            noiseSet.add(imageName)


    imageList = list(imageSet)
    noiseList = list(noiseSet)

    #imageList.extend(noiseList[: int(len(imageList) * noise)])
    random.shuffle(imageList)

    imageNum = len(imageList)

    with open(output_name, 'w+') as f:
         for x in imageList[ : int(imageNum/6)]:
             #f.write(x[:-1]+','+str(int(1))+'\n')    
             f.write(x)

    # with open(catagory + '_train.txt', 'w+') as f:
    #     for x in imageList[ : int(0.8*imageNum)]:
    #         f.write(x)

    # with open(catagory + '_val.txt', 'w+') as f:
    #     for x in imageList[int(0.8*imageNum) : int(0.9*imageNum)]:
    #         f.write(x)

    # with open(catagory + '_test.txt', 'w+') as f:
    #     for x in imageList[int(0.9*imageNum) : ]:
    #         f.write(x)


