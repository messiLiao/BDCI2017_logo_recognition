import sys
import os
import os.path
import json


def main(argv):
    if len(argv) == 1:
        json_fn = 'result/coco_result.json'
    elif len(argv) == 2:
        json_fn = argv[1]
    with open(json_fn) as fd:
        items = json.load(fd)
        total = len(items)
        right = 0
        for item in items:
            if item['image_id'] == item['category_id']:
                right += 1
        print 'rate = %f (%d/%d)' % (right*1.0/total, right, total)
    pass


if __name__ == '__main__':
    main(sys.argv)
