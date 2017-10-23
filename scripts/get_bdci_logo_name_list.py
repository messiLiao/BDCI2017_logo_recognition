# coding:utf-8
# encoding:utf-8
import os
import os.path
import sys

def get_label_list(dataset_dir):
	label_list = []
	for root, dirs, fns in os.walk(dataset_dir):
		for fn in fns:
			label, ext = os.path.splitext(fn)
			if ext.lower() in ['.jpg', '.png', '.bmp']:
				label_list.append(label)
		pass
	return label_list

def main(argv):
	
	if len(argv) == 1:
		py_file_fn = argv[0]
		dataset_dir = os.path.abspath(os.path.join(py_file_fn, "../.."))
		dataset_dir = os.path.join(dataset_dir, "dataset")
		dataset_dir = os.path.join(dataset_dir, u'LOGO图像')		
		output_name = "label.txt"
	fd = open(output_name, 'w')
	label_list = get_label_list(dataset_dir)
	for line in label_list:
		fd.write('{0}\n'.format(line.encode('utf-8')))
	fd.close()
	pass


if __name__ == '__main__':
	main(sys.argv)