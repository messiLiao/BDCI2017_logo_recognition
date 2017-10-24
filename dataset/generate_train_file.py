# coding:utf-8
#encoding:utf-8
import os
import os.path
import sys
import json
import cv2
import codecs

train_images_path = "/home/xbn/work/gitwork/BDCI2017_logo_recognition/dataset/train_images/labeld_images"
train_labels_path = '/home/xbn/work/gitwork/BDCI2017_logo_recognition/dataset/train_labels/labeld_images'
train_json_fn = '/home/xbn/work/gitwork/BDCI2017_logo_recognition/dataset/train_images/train.json'
label_json_fn = '/home/xbn/work/gitwork/BDCI2017_logo_recognition/dataset/logo_images/logos.json'
train_output_fn = '/home/xbn/work/gitwork/BDCI2017_logo_recognition/dataset/train_images/trainvalno.txt'
train_names_fn = '/home/xbn/work/gitwork/BDCI2017_logo_recognition/data/bdci2017_logo.names'

def main(argv):
    if os.path.isfile(train_json_fn):
        with open(train_json_fn) as json_file:
            train_json = json.load(json_file)
            print type(train_json)
            print train_json[0].keys()
            print train_json[0]['items']
            print train_json[0]['image_id']
            image_fn_list = []
            for item in train_json:
                image_id = item['image_id']
                fn, ext = os.path.splitext(image_id)
                image_fn = os.path.join(train_images_path, image_id)
                image_fn_list.append('{0}\n'.format(image_fn))
                image = cv2.imread(image_fn, 1)
                w, h = image.shape[:2]
                label_fn = os.path.join(train_labels_path, '{0}.txt'.format(fn))
                line_list = []
                for sub_item in item['items']:
                    sub_item_class = sub_item['label_id']
                    x1, y1, x2, y2 = sub_item['bbox']
                    cx = (x1 + x2) / 2.0 / w
                    cy = (y1 + y2) / 2.0 / h
                    cw = abs(x2 - x1) * 1.0 / w
                    ch = abs(y2 - y1) * 1.0 / h
                    line = '{0} {1} {2} {3} {4}\n'.format(sub_item_class,
                                                          cx, cy, cw, ch)

                    line_list.append(line)
                # print line_list
                with open(label_fn, 'w') as fd:
                    fd.writelines(line_list)
            with open(train_output_fn, 'w') as fd:
                fd.writelines(image_fn_list)
    if os.path.isfile(label_json_fn):
        with codecs.open(label_json_fn, 'r', 'gbk') as json_file:
            label_json = json.load(json_file)
            print type(label_json)
            print label_json[0].keys()
            print label_json[10]['image_id']
            print label_json[10]['label_id']
            label_dict = {}
            for label in label_json:
                label_id = int(label['label_id'])
                label_dict[label_id] = label['image_id']
                # print label_json[i]['image_id'], label_json[i]['label_id']
            print label_dict
            name_list = []
            for i in range(len(label_dict)):
                name_list.append(u"{0}\n".format(label_dict[i+1].replace('.jpg', '')).encode('utf-8'))
            print name_list
            with open(train_names_fn, 'w') as fd:
                fd.writelines(name_list)


        pass
    pass




if __name__ == '__main__':
    main(sys.argv)
