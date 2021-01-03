from tqdm import tqdm
import numpy as np
import os , sys
import zipfile
import random
import json
import opencv2 as cv2
import gc

def categ_switcher(category):
    categ_switcher = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13, 'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34, 'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39, 'wine glass': 40, 'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58, 'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79}
    return categ_switcher[category]

class imobj :
    '''
    Image objects for lazy evaluation
    that's a different approach from the one used in other builders
    '''
    def __init__(self,path):
        self.path = path
    def eval (self):
        return(cv2.imread(self.path))


class gp_builder:
    def __init__(self,str='coco'):
        self.data_files = ['train2017','annotations']
        self.train_dataset = []
        self.val_dataset=[]

    def load_train (self,dataset,batch_size=32, shuffle=True,
                    target_size=416, anchors=5, classes=20,
                    total_num=None, dtype=np.float32):

        total_num = len(dataset)

        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        feature_size = [x // 32 for x in target_size]
        cells = feature_size[0] * feature_size[1]


        b = 0
        while True:
            if b == 0:
                if shuffle is True:
                    idx = np.random.permutation(total_num)
                else:
                    idx = np.arange(total_num)
            if b + batch_size > total_num:
                b = 0
                yield None, None
            else:
                batch_size = batch_size

            imgs = np.zeros((batch_size,) + target_size + (3,), dtype=dtype)
            probs = np.zeros((batch_size, cells, anchors, classes), dtype=dtype)
            confs = np.zeros((batch_size, cells, anchors), dtype=dtype)
            coord = np.zeros((batch_size, cells, anchors, 4), dtype=dtype)
            proid = np.zeros((batch_size, cells, anchors, classes), dtype=dtype)
            prear = np.zeros((batch_size, cells, 4), dtype=dtype)
            areas = np.zeros((batch_size, cells, anchors), dtype=dtype)
            upleft = np.zeros((batch_size, cells, anchors, 2), dtype=dtype)
            botright = np.zeros((batch_size, cells, anchors, 2), dtype=dtype)

            box_before = {i:[] for i in range(batch_size)}
            for i in range(batch_size):
                x = dataset[idx[b+i]][0].eval() #let's just make it like this [ I know it's wrong ]
                h, w = x.shape[:2]
                cellx = 1. * w / feature_size[1]    #Number of horizontal pixels for cell
                celly = 1. * h / feature_size[0]    #Number of vertical pixels for cell

                scale_w = w/target_size[0]
                scale_h = h/target_size[1]

                #So first step you create a list of processed objects
                processed_objs = []
                for bbox in dataset[idx[b+i]][1]:
                    ul_x    = int(np.round(bbox[0]/scale_w))
                    ul_y    = int(np.round(bbox[1]/scale_h))
                    dr_x    = int(np.round(bbox[2]/scale_w))
                    dr_y    = int(np.round(bbox[3]/scale_h))
                    box_before[i].append([ul_x,ul_y,dr_x,dr_y,bbox[-1]])
                    centerx = .5 * (ul_x + dr_x)  # xmin, xmax    #W  ##Funny fact is ( W is not necessarly equal to H )
                    centery = .5 * (ul_y + dr_y)  # ymin, ymax    #H
                    cx = centerx / cellx    #ith number of cell
                    cy = centery / celly    #jth number of cell
                    if cx >= feature_size[1] or cy >= feature_size[0]:
                        continue
                    processed_objs += [[
                        int(bbox[4]),
                        cx - np.floor(cx),  # centerx   #center relative to cell boundary   #he uses it because dimensions of boxes will change so he needs something relative as hell
                        cy - np.floor(cy),  # centery   #center relative to cell boundary
                        np.sqrt(float(bbox[2] - bbox[0]) / target_size[0]),  #relative width of box , it's a ratio , hopefully it equls expo(w predicted) / 13
                        np.sqrt(float(bbox[3] - bbox[1]) / target_size[1]),  #relative height of box , it's a ratio
                        int(np.floor(cy) * feature_size[1] + np.floor(cx))  #Number of cells
                    ]]

                # Calculate placeholders' values
                for obj in processed_objs:
                    probs[i, obj[5], :, :] = [[0.] * classes] * anchors
                    probs[i, obj[5], :, obj[0]] = 1.
                    proid[i, obj[5], :, :] = [[1.] * classes] * anchors #Bullshit
                    coord[i, obj[5], :, :] = [obj[1:5]]  * anchors
                    prear[i, obj[5], 0] = obj[1] - obj[3]**2 * .5 * feature_size[1] #ul_x
                    prear[i, obj[5], 1] = obj[2] - obj[4]**2 * .5 * feature_size[0] #ul_y
                    prear[i, obj[5], 2] = obj[1] + obj[3]**2 * .5 * feature_size[1] #br_x
                    prear[i, obj[5], 3] = obj[2] + obj[4]**2 * .5 * feature_size[0] #br_y
                    confs[i, obj[5], :] = [1.] * anchors
                # Finalise the placeholders' values
                ul = np.expand_dims(prear[i, :, 0:2], 1)    #(X,Y)
                br = np.expand_dims(prear[i, :, 2:4], 1)
                wh = br - ul
                area = wh[:, :, 0] * wh[:, :, 1]
                upleft[i, :, :, :] = np.concatenate([ul] * anchors, 1)
                botright[i, :, :, :] = np.concatenate([br] * anchors, 1)
                areas[i, :, :] = np.concatenate([area] * anchors, 1)

                imgs[i] = cv2.resize(x, target_size,
                                     interpolation=cv2.INTER_LINEAR)
            yield imgs, [probs, confs, coord, proid, areas, upleft, botright],box_before
            b += batch_size


    def set_data(self):
        self.prepare()

    def get_andata(self):
        print(self.data_files)
        annotations_dirs = [self.data_files[1]+'/instances_train2014.json']#,self.data_files[1]+'/instances_val2014.json']
        train_andata = dict()

        for itr , annotation_dir in enumerate(annotations_dirs):
            with open(annotation_dir,'r') as json_file:
                train_andata=json.load(json_file)
               
        return train_andata

    #A function that returns the label ( class / category )of the current bounding box
    def get_cat (self,id,andata):
        return next(item["name"] for item in andata["categories"] if item["id"]==id)

    def get_bbox(self,train_andata):
        timg_bbox = dict()

        for itr,andata in enumerate([train_andata]):
            img_bbox = dict()
            for annotation in andata['annotations']:
                current_id = annotation["image_id"]

                ul_x = annotation["bbox"][0]
                ul_y = annotation["bbox"][1]
                dr_x = annotation["bbox"][0] + annotation["bbox"][2]
                dr_y = annotation["bbox"][1] + annotation["bbox"][3]

                if current_id in img_bbox.keys():
                    img_bbox[current_id].append([ul_x,ul_y,dr_x,dr_y,categ_switcher(self.get_cat(annotation["category_id"],andata))])
                else :
                    img_bbox[current_id] = [[ul_x,ul_y,dr_x,dr_y,categ_switcher(self.get_cat(annotation["category_id"],andata))]]

            timg_bbox = img_bbox.copy()
            
        return timg_bbox

    def get_key(self,string):

        '''
        this function returns the image id from image file stored in the disk
        if image file is stored as ' 000546.png'
        image ID will be 546
        and that what hopefully this function does.
        '''
        string =string.split('.')[0].split('_')[-1]
        for i,j in enumerate(string):
            if(j!='0'):
                break
        return int(string[i:])

    def get_imgs(self):
        imgs_dir = [self.data_files[0]]
        timgs=dict()
        corrupted_keys=[]

        for itr,direc in enumerate(imgs_dir) :
            imdict=dict()
            all_images_names = [img for img in os.listdir(direc) if os.path.isfile(os.path.join(direc,img))]

            for image in tqdm(all_images_names):
                key = self.get_key(image)
                try:
                    im = imobj(os.path.join(direc,image))
                    imdict[key]=im
                except :
                    corrupted_keys.append(key)
                    continue

            timgs = imdict
            

        return timgs

    def prepare(self):
        train_andata = self.get_andata()
        timg_bbox    = self.get_bbox(train_andata)
        timgs        =self.get_imgs()

        for key in sorted(timgs.keys()):
            try:
                self.train_dataset.append([timgs[key],np.array(timg_bbox[key])])
            except :continue

        random.shuffle(self.train_dataset)

        self.report(timgs,timg_bbox)

    def report(self,timgs,timg_bbox):
        first = sorted(timg_bbox.keys())
        second= sorted(timgs.keys())
        found =0
        not_found=0
        for keyi in second:
          try :
            opn = timg_bbox[keyi]
            opn2= timgs[keyi]
            found+=1
          except :
            not_found +=1
        print('Succeful engagement : {} , Unsecceful engagement : {}'.format(found,not_found))
        print('length of read imags : {} , length of annotations corresponding to images {}'.format(len(first),len(second)))
        print('length of downloaded images : {}'.format(len(os.listdir('train2014'))))

    def _get_data(self):
        return np.array(self.train_dataset)

    def _get_traindata(self):
        return np.array(self.train_dataset)
