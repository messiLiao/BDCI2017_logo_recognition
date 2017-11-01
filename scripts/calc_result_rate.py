# codeing:utf-8
# encoding:utf-8
import os
import cv2
import os.path
import sys
import json
import codecs
import gzip
import tarfile
import random

show_enable = True
def show_result(result_fn):
	global show_enable
	dataset_root = os.path.join("G:", "datasets", "datafountain", "dataset")
	train_images_dir = os.path.join("train_images", "labeld_images")
	image_path = os.path.join(dataset_root, train_images_dir)

	image_dict = dict([])
	skip = True
	show_right_enable = True
	with open(result_fn, 'r') as fd:
		lines = fd.readlines()
		# print len(lines)
		for line in lines:
			image_id, prob, x1, y1, x2, y2 = line.split(' ')[:6]
			try:
				prob, x1, y1, x2, y2 = (prob), float(x1), float(y1), float(x2), float(y2)
			except:
				print "error: ", result_fn, prob, x1, y1, x2, y2
			if image_id not in image_dict:
				image_dict[image_id] = []
			image_dict[image_id].append((prob, x1, y1, x2, y2))
	# for image_id, (prob, x1, y1, x2, y2) in image_dict.items():
	train_labels = None
	right = 0
	total = len(image_dict)
	names = os.path.splitext(os.path.basename(result_fn))[0].split("_")
	label_name = names[-1].replace("1", "")
	right_label = -1
	label_dict = dict({})
	name_dict = dict({})
	with codecs.open("../results/id_label.json", 'r', 'utf-8') as fd:
		try:
			label_dict = json.load(fd)
		except:
			print "wrong decoder:", result_fn
		for _id, _name in label_dict.items():
			name_dict[_name] = int(_id)

	right_label = name_dict[label_name.decode("utf-8")]
	if right_label < 0:
		print "-=-=-=", name_dict[label_name.decode("utf-8")]
		print "right label not found"
	else:
		print "label_name is %s, label id is %d" % (label_name, right_label)

	with codecs.open("../results/train.json", 'r', 'utf-8') as fd:
		train_labels = json.load(fd)
		for x in train_labels:
			print x
			break

	for image_id, boxes in image_dict.items():
		if show_enable and (not skip or show_right_enable):
			image = cv2.imread(os.path.join(image_path, "%s.jpg" % image_id))
			for (prob, x1, y1, x2, y2) in boxes:
				x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
				print (x1, y1), (x2, y2)
				cv2.rectangle(image, (x1, y1), (x2, y2), (150, 233, 100), 2)

		
		items = [label for label in train_labels if label['image_id'].replace('.jpg', '') == image_id]
		image_is_right = False
		for image_item in items:
			for box_item in image_item['items']:
				if int(box_item["label_id"]) == right_label:
					right += 1
					if show_enable and show_right_enable:
						x1, y1, x2, y2 = box_item['bbox']
						image_is_right = True
						cv2.rectangle(image, (x1, y1), (x2, y2), (211, 20, 100), 2)
				if show_enable and not skip:
					x1, y1, x2, y2 = box_item['bbox']
					cv2.rectangle(image, (x1, y1), (x2, y2), (211, 20, 100), 2)

		if show_enable and (not skip or (show_right_enable and image_is_right)):
			cv2.imshow("geely_image", image)
			key = cv2.waitKey(0) & 255
			if key in [ord('q'), ord('Q')]:
				show_enable = False
			elif key in [ord('s'), ord('S')]:
				skip = True
				show_right_enable = False
	print result_fn, ":rates: %d/%d \n" %(right, total)
	return right, total


def classify(result_fn, label):
	image_dict = dict([])
	with open(result_fn, 'r') as fd:
		lines = fd.readlines()
		print len(lines)
		for line in lines:
			image_id, prob, x, y, w, h = line.split(' ')[:6]
			prob, x, y, w, h = (prob), float(x), float(y), float(w), float(h)
			image_dict[image_id] = 1

	total = len(image_dict)
	print total

	right_class = []
	with codecs.open("../results/train.json", 'r', 'gbk') as fd:
		right = 0
		labels = json.load(fd)
		for _label in labels:
			fn = _label['image_id'].replace('.jpg', '')
			for item in _label['items']:
				if int(item["label_id"]) == label and fn in image_dict:
					# print fn, item["label_id"]
					right_class.append(fn)
	right = len(right_class)
	print right
	pass
def main(argv):
	if len(argv) == 1:
		image_id = '500_wKgFVVl1fv2ALsMsAAeEj_F9MK8417.jpg'
		image_id = '500_wKgFU1ly--GAMqDXAAHiykEtOoo557.jpg'
	elif len(argv) == 2:
		image_id = argv[1]
	image_fn = os.path.join(u'./', image_id)
	image = cv2.imread(image_fn, 1)
	print image.shape[:2]
	json_fn = '../train.json'
	with open(json_fn, 'r') as fd:
		json_data = json.load(fd)
		for items in json_data:
			if items['image_id'] == image_id:
				for i, item in enumerate(items['items']):
					x1, y1, x2, y2 = item['bbox']
					sub_image = image[y1:y2, x1:x2]
					print sub_image.shape, item
					cv2.imshow("sub_%d" % i, sub_image)
					cv2.waitKey(0)
	pass

def cross_rect_area(p1_tl, p1_br, p2_tl, p2_br):
	w1 = abs((p1_tl[0] - p1_br[0]))
	h1 = abs((p1_tl[1] - p1_br[1]))
	w2 = abs((p2_tl[0] - p2_br[0]))
	h2 = abs((p2_tl[1] - p2_br[1]))
	x_min = min((p1_tl[0], p1_br[0], p2_tl[0], p2_br[0]))
	x_max = max((p1_tl[0], p1_br[0], p2_tl[0], p2_br[0]))
	y_min = min((p1_tl[1], p1_br[1], p2_tl[1], p2_br[1]))
	y_max = max((p1_tl[1], p1_br[1], p2_tl[1], p2_br[1]))
	if (x_max - x_min) <= (w1 + w2):
		overlap_w = (w1 + w2) - (x_max - x_min)
	else:
		overlap_w = 0

	if (y_max - y_min) <= (h1 + h2):
		overlap_h = (h1 + h2) - (y_max - y_min)
	else:
		overlap_h = 0
	overlap_s = overlap_w * overlap_h
	s1 = w1 * h1
	s2 = w2 * h2
	rate = overlap_s * 1.0 / (s1 + s2 - overlap_s)
	return rate
	pass

def show_test_result(test_result_fn_list, threshold, image_show_enable, print_enable, save_to_file):
	result_dict = dict({})
	output_filename = "preliminary_result.json"

	for test_result_fn in test_result_fn_list:
		result_list = []
		with open(test_result_fn, 'r') as fd:
			result_list = fd.readlines()

		no_object_count = 0
		for line in result_list:
			if line.find('Unable') >= 0:
				continue
			if line.find("skip") >= 0:
				continue
			fn, class_id, prob, x1, y1, x2, y2 = line.split(' ')[:7]
			class_id, prob, x1, y1, x2, y2 = int(class_id), float(prob), int(x1), int(y1), int(x2), int(y2)
			# print fn, class_id, prob, x1, y1, x2, y2
			_, image_fn = os.path.split(fn)
			# print image_fn
			if image_fn not in result_dict:
				result_dict[image_fn] = []
			result_dict[image_fn].append((class_id, prob, x1, y1, x2, y2))

		# break
	print 'image count=', len(result_dict)
	skip_cnt = random.randint(0, 10)

	label_dict = dict({})
	with codecs.open("../results/id_label.json", 'r', 'utf-8') as fd:
		try:
			label_dict = json.load(fd)
		except:
			print "wrong decoder:", result_fn

	output_json_list = []
	for image_fn, items in result_dict.iteritems():
		# if skip_cnt > 0:
		# 	skip_cnt -= 1
		# 	continue
		# skip_cnt = random.randint(0, 10)
		full_image_fn = os.path.join("G:", "datasets", "datafountain", "dataset", "preliminary_images", image_fn)
		
		if print_enable:
			print "=========", image_fn, "=="
			print "items:", items
		items = [(class_id, prob, x1, y1, x2, y2) for class_id, prob, x1, y1, x2, y2 in items if prob > threshold]
		if len(items) < 1:
			no_object_count += 1
			continue
		new_items = []
		
		if print_enable:
			print "items:", items
		new_items.append([items.pop()])
		while len(items) > 0:
			class_idi, probi, xi1, yi1, xi2, yi2 = items.pop()
			overlap_rect_index_list = []
			for i, sub_item_list in enumerate(new_items):
				for class_idj, probj, xj1, yj1, xj2, yj2 in sub_item_list:
					overlap_rate = cross_rect_area((xi1, yi1), (xi2, yi2), (xj1, yj1), (xj2, yj2))					
					if print_enable:
						print "overlap_rate=", overlap_rate
					if overlap_rate > 0.4:
						overlap_rect_index_list.append((i, overlap_rate))
			if len(overlap_rect_index_list) > 0:
				overlap_rect_index_list.sort(key=lambda x:x[1])
				new_items[overlap_rect_index_list[0][0]].append((class_idi, probi, xi1, yi1, xi2, yi2))
			else:
				new_items.append([(class_idi, probi, xi1, yi1, xi2, yi2)])

		
		if print_enable:
			print "new_items:", new_items
		items = []
		for l in new_items:
			l.sort(key=lambda x:x[1])
			items.append(l[-1])
		
		if print_enable:
			print "no overlap items:", items
		if image_show_enable:
			image = cv2.imread(full_image_fn, 1)
			for class_id, prob, x1, y1, x2, y2 in items:
				cv2.rectangle(image, (x1, y1), (x2, y2), (211, 20, 100), 2)
				text_id = "%d" % class_id
				text_label = label_dict["%04d" % class_id]
				cv2.putText(image, text_id, (x2, y1), cv2.cv.CV_FONT_HERSHEY_DUPLEX, 1.0,(211, 200, 100));
				
				if print_enable:
					print text_id, "->", text_label, prob
			cv2.imshow("pre", image)
			key = cv2.waitKey(0) & 0xFF
			if key in [ord('q'), ord('Q')]:
				image_show_enable = False
		if save_to_file:
			items_dict = dict({})
			items_dict["image_id"] = image_fn
			items_dict["type"] = "A"
			items_dict["items"] = []
			for class_id, prob, x1, y1, x2, y2 in items:
				sub_item_dict = dict({})
				sub_item_dict['label_id'] = "%04d" % class_id
				sub_item_dict['bbox'] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
				sub_item_dict['score'] = "%.6f" % prob
				items_dict["items"].append(sub_item_dict)
			output_json_list.append(items_dict)
	if save_to_file:
		with open(output_filename, 'w') as fd:
			json.dump(output_json_list, fd)
		with open(output_filename, 'r') as fd:
			llist = json.load(fd)
			print len(llist), no_object_count

if __name__ == '__main__':
	# classify('../results/geely.txt', 9)
	file_list = os.listdir("../results")
	file_list = [fn for fn in file_list if fn.endswith(".txt")]
	right = 0
	total = 1
	for i, file_name in enumerate(file_list):
		break
		r, t = show_result(os.path.join("..", "results", file_name))
		right += r
		total += t
	print "total rates: %f ( %d / %d)" % (right * 1.0 / total, right, total)
	# main(sys.argv)
	# image = cv2.imread("G:/datasets/datafountain/dataset/preliminary_images/500_00CO_eYEwzzvk_mzt0g9LM.jpg", 1)
	# x1, x2, y1, y2 = 227, 258, 167, 190
	# cv2.rectangle(image, (x1, y1), (x2, y2), (211, 20, 100), 2)
	# cv2.imshow("pre", image)
	# cv2.imshow("sub", sub_image)
	# cv2.waitKey(0)
	save_to_file = True
	image_show_enable = False
	print_enable = False
	threshold = 0.31
	result_file_list = ["../preliminary_predict_3.txt", "../preliminary_predict_2.txt", "../preliminary_predict.txt"]
	show_test_result(result_file_list, threshold, image_show_enable, print_enable, save_to_file)
	p1_tl, p1_br, p2_tl, p2_br = (287, 104), (351, 169), (291, 106), (342, 167)
	# print cross_rect_area((4, 1.2), (0, 0), (1, 1), (3, 3))
	# print cross_rect_area(p1_tl, p1_br, p2_tl, p2_br)