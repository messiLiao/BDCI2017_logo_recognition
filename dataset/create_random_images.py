#coding:utf-8
#ecoding:utf-8
import os
import cv2
import numpy as np
import random
import multiprocessing

def show_mask(image_dir):
    image_fn_list = os.listdir(image_dir)
    print len(image_fn_list)
    bmp_index = 0;
    for image_fn in image_fn_list:
        if not image_fn.endswith('.bmp'):
            continue
        bmp_index += 1
        if bmp_index < 31:
            continue
        full_fn = os.path.join(image_dir, image_fn)
        image = cv2.imread(full_fn, 0)
        cv2.imshow("gray", image)
        mask = np.zeros_like(image)
        if image_fn == 'jeep1.bmp':
            mask[image == 0] = 255
            kernel = np.ones((3, 3), dtype=np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = 255 - mask
        else:
            mask[image == 255] = 255
            kernel = np.ones((3, 3), dtype=np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # cv2.imwrite(os.path.join("logos_mask", image_fn), mask)
        cv2.imshow("mask", mask)
        key = cv2.waitKey(0) & 255
        if key in [ord('q'), ord('Q'), 27]:
            break
        else:
            print key

        print image_fn

def region_select(image_fn):
    def mouse_callback(event,x,y,flags,param):
        if event == cv2.EVENT_FLAG_LBUTTON:
            last_point[0] = x
            last_point[1] = y
            print "EVENT_FLAG_LBUTTON", last_point, id(last_point)
            rectangle[2:4] = last_point
            print "%d, %d, %d, %d" % (rectangle[0], rectangle[1], rectangle[2], rectangle[3])
            pass
        elif event ==cv2.EVENT_LBUTTONDOWN:
            print "EVENT_LBUTTONDOWN", x, y
        elif event ==cv2.EVENT_MOUSEMOVE:
            if last_point is not None:
                pass
            tmp_image = np.copy(image)    
            cv2.rectangle(tmp_image, (last_point[0], last_point[1]), (x, y), (0, 255, 0), 2)
            rectangle[:] = last_point[0], last_point[1], x, y
            cv2.imshow("origin", tmp_image)
    image = cv2.imread(image_fn)
    last_point = np.array((0, 0), dtype=np.uint32)
    rectangle = np.array((0, 0, 0, 0), dtype=np.uint32)
    print id(last_point)
    cv2.imshow("origin", image)
    cv2.setMouseCallback('origin', mouse_callback)
    cv2.waitKey(0)
    return rectangle

def cut_image_and_mask(image_fn, mask_fn, rectangle):
    origin_image = cv2.imread(image_fn)
    mask_image = cv2.imread(mask_fn)
    x_, y_, _x, _y = rectangle
    x_, y_, _x, _y = min(x_, _x), min(y_, _y), max(x_, _x), max(y_, _y)

    print y_, _y, x_, _x
    origin = origin_image[y_:_y, x_:_x, :]
    mask = mask_image[y_:_y, x_:_x, :]
    cv2.imshow("origin", origin_image[y_:_y, x_:_x, :])
    cv2.imshow("mask", mask_image[y_:_y, x_:_x, :])
    key = cv2.waitKey(0) & 255
    if key in [ord('\n'), ord('\r')]:
        print "Yes"
        cv2.imwrite(image_fn, origin)
        cv2.imwrite(mask_fn, mask)

def crop_image():
    # region_select('./logos/beiqi.jpg')
    # cut_image_and_mask('./logos/beiqi.jpg', './logos_mask/beiqi.bmp', (374, 205, 134, 18))
    
    # region_select('./logos/ford.jpg')
    # cut_image_and_mask('./logos/ford.jpg', './logos_mask/ford.bmp', (1019,  727, 2, 297))

    # region_select('./logos/jeep.jpg')
    # cut_image_and_mask('./logos/jeep.jpg', './logos_mask/jeep.bmp', (314, 162, 59, 40))

    # region_select('./logos/posche.jpg')
    # cut_image_and_mask('./logos/posche.jpg', './logos_mask/posche.bmp', (439, 418, 155, 103))
    
    # region_select('./logos/mazida.jpg')
    # cut_image_and_mask('./logos/mazida.jpg', './logos_mask/mazida.bmp', (452, 380, 126, 105))
    
    # region_select('./logos/kia.jpg')
    # cut_image_and_mask('./logos/kia.jpg', './logos_mask/kia.bmp', (496, 328, 114, 101))
    
    # region_select('./logos/geely.jpg')
    # cut_image_and_mask('./logos/geely.jpg', './logos_mask/geely.bmp', (629, 532, 126, 43))
    
    # region_select('./logos/mg.jpg')
    # cut_image_and_mask('./logos/mg.jpg', './logos_mask/mg.bmp', (436, 330, 159, 60))
    
    # region_select('./logos/linken.jpg')
    # cut_image_and_mask('./logos/linken.jpg', './logos_mask/linken.bmp', (219, 252, 65, 12))
    
    # region_select('./logos/skoda.jpg')
    # cut_image_and_mask('./logos/skoda.jpg', './logos_mask/skoda.bmp', (188, 214, 9, 53))
    pass

def scale_image(image, mask, min_side, max_side):
    scale_w = random.randint(min_side, max_side)
    scale_h = random.randint(min_side, max_side)
    image = cv2.resize(image, (scale_h, scale_w))
    mask = cv2.resize(mask, (scale_h, scale_w))
    return image, mask
    pass

def erode_or_dilate_mask(mask, min_kernel_size, max_kernel_size):
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    if random.random() <= 0.5:
        function = cv2.erode
    else:
        function = cv2.dilate
    mask = function(mask, kernel)
    return mask

def contrast(image, min_rate, max_rate):
    pass

def perspective(image, mask, min_side, max_side):
    h, w = mask.shape
    nh, nw = random.randint(min_side, max_side), random.randint(min_side, max_side)
    img_points = np.array(((0, 0), (0, h), (w, 0), (w, h)), dtype=np.float32)
    x = 3
    per_points = np.array(((random.randint(0, int(nw/x)), random.randint(0, int(nh/x))),\
                  (random.randint(0, int(nw/x)), random.randint(int(nh/x), nh)),\
                  (random.randint(int(nw*(x-1)/x), nw), random.randint(0, int(nh/x))),\
                  (random.randint(int(nw*(x-1)/x), nw), random.randint(int(nh*(x-1)/x), nh))), dtype=np.float32)
    trans_mat = cv2.getPerspectiveTransform(img_points, per_points)
    image = cv2.warpPerspective(image,
                                trans_mat, (nw, nh)) # cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    mask = cv2.warpPerspective(mask,
                               trans_mat, (nw, nh))
    return image, mask
    

def exposure_image(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    print hsv.shape
    h, s, v = cv2.split(hsv)
    if random.random() < 0.5:
        rate = (random.random() + 1) * (256 / np.max(v))
    else:
        rate = random.random()
    nv = v * rate
    nv[nv > 255] = 255    
    v = nv.astype(np.uint8)

    if random.random() < 0.5:
        rate = (random.random() + 1) * (256 / np.max(v))
    else:
        rate = random.random()
    ns = s * rate
    ns[ns > 255] = 255    
    s = ns.astype(np.uint8)


    hsv = cv2.merge((h, s, v))

    image = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return image

def merge_background(image, mask, bg_image):
    h, w = bg_image.shape[:2]
    mh, mw = mask.shape[:2]
    pos_x = random.randint(0, w - mw)
    pos_y = random.randint(0, h - mh)

    sub_bg_image = bg_image[pos_y:pos_y+mh, pos_x:pos_x+mw]
    index = mask > 0
    sub_bg_image[mask > 0] = (0, 0, 0)
    image[mask == 0] = (0, 0, 0)
    sub_bg_image += image
    bbox = (pos_y, pos_x, mh+pos_y, mw+pos_x)
    return bg_image, bbox
    pass

def get_random_alpha_str(length):
    s = []
    for i in range(length):
        index = random.randint(0, 26*2+10-1)
        if index < 26:
            c = chr(ord('a') + index)
        elif index < 52:
            c = chr(ord('A') + index - 26)
        else:
            c = chr(ord('0') + index - 52)
        s.append(c)
    return ''.join(s)


def SynthProcess(image_fn, mask_fn, bg_image_fn, output_dir, class_id):    

    min_side = 30
    max_side = 100
    min_kernel_size = 1
    max_kernel_size = 15

    image_output_name = "synth_images"
    label_output_name = "synth_labels"

    image_output_dir = os.path.join(output_dir, image_output_name)
    label_output_dir = os.path.join(output_dir, label_output_name)
    image = cv2.imread(image_fn)
    mask = cv2.imread(mask_fn, 0)
    bg_image = cv2.imread(bg_image_fn)

    if (image is None) or (mask is None) or (bg_image is None):
        return None

    mask = erode_or_dilate_mask(mask, min_kernel_size, max_kernel_size)

    image, mask = perspective(image, mask, min_side, max_side)

    image = exposure_image(image)

    output_image, bbox = merge_background(image, mask, bg_image)

    _, fn = os.path.split(bg_image_fn)
    name, ext = os.path.splitext(fn)

    random_alpha_str = get_random_alpha_str(8)
    output_image_fn = os.path.join(image_output_dir, '%s_%s%s' % (name, random_alpha_str, ext))
    output_label_fn = os.path.join(label_output_dir, '%s_%s.txt' % (name, random_alpha_str))

    result = "%s %s\n" % (os.path.join(image_output_name, os.path.split(output_image_fn)[1]),
                                        os.path.join(label_output_name, os.path.split(output_label_fn)[1]))

    cv2.imwrite(output_image_fn, output_image)
    line = None
    with open(output_label_fn, 'w') as fd:
        line = "%s %d " % (os.path.split(output_image_fn)[1], class_id)
        line += "%d %d %d %d" % bbox
        fd.write(line)
    return result

def generate_syntm_images(image_dir, mask_dir, background_dir, output_dir, generate_cnt, debug=False):
    label_class_id = dict()
    label_class_id['beiqi'] = 1
    label_class_id['ford'] = 2
    label_class_id['skoda'] = 3
    label_class_id['qichen'] = 4
    label_class_id['honda'] = 5
    label_class_id['nissan'] = 6
    label_class_id['cadlock'] = 7
    label_class_id['linmu'] = 8
    label_class_id['geely'] = 9
    label_class_id['geely1'] = 9
    label_class_id['posche'] = 10
    label_class_id['jeep'] = 11
    label_class_id['jeep1'] = 11
    label_class_id['baojun'] = 12
    label_class_id['ronwei'] = 13
    label_class_id['linken'] = 14
    label_class_id['toyota'] = 15
    label_class_id['buik'] = 16
    label_class_id['qirui'] = 17
    label_class_id['kia'] = 18
    label_class_id['haval'] = 19
    label_class_id['audi'] = 20
    label_class_id['randrover'] = 21
    label_class_id['dasauto'] = 22
    label_class_id['chuanqi'] = 23
    label_class_id['changan'] = 24
    label_class_id['mg'] = 25
    label_class_id['leno'] = 26
    label_class_id['lexs'] = 27
    label_class_id['bmw'] = 28
    label_class_id['mazida'] = 29
    label_class_id['benz'] = 30

    image_fn_list = os.listdir(image_dir)
    bg_image_fn_list = os.listdir(background_dir)

    image_cnt = 0

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpu_count)
    print "cpu_count = %d" % cpu_count

    image_output_name = "synth_images"
    label_output_name = "synth_labels"

    image_output_dir = os.path.join(output_dir, image_output_name)
    label_output_dir = os.path.join(output_dir, label_output_name)
    
    if not os.path.isdir(image_output_dir):
        os.mkdir(image_output_dir)
    if not os.path.isdir(label_output_dir):
        os.mkdir(label_output_dir)

    list_output_fn = os.path.join(output_dir, "synth_logo_list.txt")
    list_output_fd = open(list_output_fn, 'w')
    while image_cnt < generate_cnt:
        image_fn = random.choice(image_fn_list)

        name, ext = os.path.splitext(image_fn)
        image_fn = os.path.join(image_dir, image_fn)
        mask_fn = os.path.join(mask_dir, "%s.bmp" % name)

        class_id = label_class_id[name]


        bg_image_fn = random.choice(bg_image_fn_list)
        bg_image_fn = os.path.join(background_dir, bg_image_fn)
        
        result = pool.apply_async(SynthProcess, (image_fn, mask_fn, bg_image_fn, output_dir, class_id))
        line = result.get()
        if line is not None:
            list_output_fd.write(line)
        if image_cnt % 1000 == 0:
            print "image count = %6d." % image_cnt
        image_cnt += 1

    pool.close()
    pool.join()
    list_output_fd.close()
    pass

def main(image_dir, mask_dir, background_dir, output_dir, generate_cnt, debug=False):
    label_class_id = dict()
    label_class_id['beiqi'] = 1
    label_class_id['ford'] = 2
    label_class_id['skoda'] = 3
    label_class_id['qichen'] = 4
    label_class_id['honda'] = 5
    label_class_id['nissan'] = 6
    label_class_id['cadlock'] = 7
    label_class_id['linmu'] = 8
    label_class_id['geely'] = 9
    label_class_id['geely1'] = 9
    label_class_id['posche'] = 10
    label_class_id['jeep'] = 11
    label_class_id['jeep1'] = 11
    label_class_id['baojun'] = 12
    label_class_id['ronwei'] = 13
    label_class_id['linken'] = 14
    label_class_id['toyota'] = 15
    label_class_id['buik'] = 16
    label_class_id['qirui'] = 17
    label_class_id['kia'] = 18
    label_class_id['haval'] = 19
    label_class_id['audi'] = 20
    label_class_id['randrover'] = 21
    label_class_id['dasauto'] = 22
    label_class_id['chuanqi'] = 23
    label_class_id['changan'] = 24
    label_class_id['mg'] = 25
    label_class_id['leno'] = 26
    label_class_id['lexs'] = 27
    label_class_id['bmw'] = 28
    label_class_id['mazida'] = 29
    label_class_id['benz'] = 30

    image_output_name = "synth_images"
    label_output_name = "synth_labels"
    image_output_dir = os.path.join(output_dir, image_output_name)
    label_output_dir = os.path.join(output_dir, label_output_name)
    list_output_fn = os.path.join(output_dir, "synth_logo_list.txt")

    image_fn_list = os.listdir(image_dir)
    bg_image_fn_list = os.listdir(background_dir)

    min_side = 30
    max_side = 100
    min_kernel_size = 1
    max_kernel_size = 15
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(image_output_dir):
        os.mkdir(image_output_dir)
    if not os.path.isdir(label_output_dir):
        os.mkdir(label_output_dir)
    image_cnt = 0
    list_output_fd = open(list_output_fn, 'w')
    while image_cnt < generate_cnt:
        image_cnt += 1
        # for image_fn in image_fn_list:
        image_fn = random.choice(image_fn_list)
        name, ext = os.path.splitext(image_fn)
        image_fn = os.path.join(image_dir, image_fn)
        mask_fn = os.path.join(mask_dir, "%s.bmp" % name)
        class_id = label_class_id[name]
        print image_fn, mask_fn
        image = cv2.imread(image_fn)
        mask = cv2.imread(mask_fn, 0)
        mask = erode_or_dilate_mask(mask, min_kernel_size, max_kernel_size)

        # image, mask = scale_image(image, mask, min_side, max_side)
        image, mask = perspective(image, mask, min_side, max_side)
        image = exposure_image(image)
        if debug:
            cv2.imshow("image", image)
            cv2.imshow("mask", mask)

        bg_image_fn = random.choice(bg_image_fn_list)
        name, ext = os.path.splitext(bg_image_fn)
        random_alpha_str = get_random_alpha_str(8)
        output_image_fn = os.path.join(image_output_dir, '%s_%s%s' % (name, random_alpha_str, ext))
        output_label_fn = os.path.join(label_output_dir, '%s_%s.txt' % (name, random_alpha_str))
        bg_image_fn = os.path.join(background_dir, bg_image_fn)
        bg_image = cv2.imread(bg_image_fn)
        output_image, bbox = merge_background(image, mask, bg_image)
        print bbox, class_id
        cv2.imwrite(output_image_fn, output_image)
        if debug:
            cv2.imshow("image", image)
            cv2.imshow("mask", mask)
            cv2.imshow("bg_image", output_image)
            key = cv2.waitKey(0) & 255
            if key in [ord('q'), 23]:
                break
        with open(output_label_fn, 'w') as fd:
            line = "%s %d " % (os.path.split(output_image_fn)[1], class_id)
            line += "%d %d %d %d" % bbox
            fd.write(line)
        list_output_fd.write("%s %s\n" % (os.path.join(image_output_name, os.path.split(output_image_fn)[1]),
                                        os.path.join(label_output_name, os.path.split(output_label_fn)[1])))
    list_output_fd.close()
    print "output files count:", generate_cnt
    pass


if __name__ == '__main__':
    image_dir = 'logos'
    mask_dir = 'logos_mask'
    background_dir = 'background_images'
    output_dir = '.'
    generate_cnt = 10

    image_dir = './logos'
    mask_dir = './logos_mask'
    background_dir = '/home/xbn/work/data/background_images'
    output_dir = '/home/xbn/work/kongchang/data'
    generate_cnt = 10


    # main(image_dir, mask_dir, background_dir, output_dir, generate_cnt, debug=True)
    generate_syntm_images(image_dir, mask_dir, background_dir, output_dir, generate_cnt, debug=True)
    