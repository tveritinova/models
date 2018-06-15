import os

import tensorflow as tf
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

import csv

data = {}

image_width = 1280.0
image_height = 720.0

LABELS = set()

# for i in range(3,4):
#     for name in ['test', 'train']:
        #with open('/jet/prs/russian_signs/rtsd-d'+str(i)+'-gt/'+name+'_filenames.txt') as f:
            #print("opened", '/jet/prs/russian_signs/rtsd-d'+str(i)+'-gt/'+name+'_filenames.txt')
#             for filename in f:
#                 filename = 'rtsd-d'+str(i)+'-frames/'+name+'/'+filename[:-1]

#                 data[filename] = {}
#                 data[filename]['xmins'] = []
#                 data[filename]['xmaxs'] = []
#                 data[filename]['ymins'] = []
#                 data[filename]['ymaxs'] = []
#                 data[filename]['classes_text'] = []
        
#     for f in ['danger', 'mandatory', 'prohibitory']:
#         print(f)
#         for name in ['test', 'train']:
#             with open('/jet/prs/russian_signs/rtsd-d'+str(i)+'-gt/'+f+'/'+name+'_gt.csv', 'r') as csvfile:

for filename in os.listdir('/jet/prs/russian_signs/rtsd/rtsd-frames'):
    data[filename] = {}
    data[filename]['xmins'] = []
    data[filename]['xmaxs'] = []
    data[filename]['ymins'] = []
    data[filename]['ymaxs'] = []
    data[filename]['classes_text'] = []   
    data[filename]['classes'] = []
    
larger_cnt = 0
boxes_cnt = 0

with open('/jet/prs/russian_signs/rtsd/full-gt.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the headers
    for row in reader:
        filename, x_from, y_from, w, h, text, label = row
        LABELS.add(text)

        #filename = 'rtsd-d'+str(i)+'-frames/'+name+'/'+filename
        
        boxes_cnt += 1
        
        if float(x_from) + float(w) > image_width or float(y_from) + float(h) > image_height:
            larger_cnt += 1
            continue;
            
        data[filename]['xmins'].append(float(x_from) / image_width)
        data[filename]['xmaxs'].append((float(x_from) + float(w)) / image_width)
        data[filename]['ymins'].append(float(y_from) / image_height)
        data[filename]['ymaxs'].append((float(y_from) + float(h)) / image_height)
        data[filename]['classes_text'].append(text)   
        data[filename]['classes'].append(int(label))

writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

LABELS = list(LABELS)
print("classes cnt:", len(LABELS))
print("images cnt:", len(data.keys()))
print("boxes:", boxes_cnt)
print("invalid boxes:", larger_cnt)

cl = ['1_13', '7_4', '7_3', '7_7', '5_22', '2_3', '3_13_r4.3', '1_12', '4_1_5', '8_5_4', '4_2_2', '1_1', '2_3_3', '3_12_n6', '8_5_2', '1_22', '8_2_1', '3_14_r3.5', '3_13_r5.2', '1_8', '8_3_2', '3_27', '3_13_r4.5', '3_24_n80', '3_12_n3', '5_15_1', '2_2', '3_11_n9', '3_11_n13', '1_17', '8_6_2', '3_13_r4', '1_11', '4_1_6', '3_13_r3', '1_19', '2_4', '5_6', '2_6', '7_11', '3_28', '8_8', '3_24_n70', '3_4_n5', '5_15_7', '8_4_4', '3_31', '8_6_4', '6_2_n20', '4_1_1', '3_25_n20', '3_2', '3_14_r3.7', '8_2_3', '1_14', '8_17', '3_13_r3.9', '1_10', '5_15_3', '5_5', '4_1_2_1', '1_11_1', '2_3_6', '6_8_3', '3_13_r3.5', '8_23', '3_13_r4.2', '3_11_n23', '6_6', '5_15_2', '5_17', '7_14', '5_16', '3_1', '8_1_3', '8_3_1', '1_7', '3_16_n1', '3_21', '8_13', '5_11', '3_18_2', '3_30', '3_13_r3.7', '3_4_n8', '3_16_n3', '5_3', '7_18', '6_2_n60', '4_1_2_2', '2_3_4', '3_4_n2', '8_15', '3_12_n10', '1_20', '1_15', '4_8_3', '1_23', '1_31', '5_15_5', '8_2_2', '1_20_2', '6_3_1', '2_3_5', '8_4_1', '6_15_2', '3_13_r4.1', '1_5', '3_6', '8_16', '4_8_2', '3_24_n10', '5_21', '2_1', '3_20', '3_14_r2.7', '7_5', '1_12_2', '5_20', '1_21', '6_8_1', '4_3', '1_25', '3_14_r3', '3_25_n40', '7_6', '3_32', '3_25_n50', '3_24_n30', '4_5', '8_18', '7_1', '3_13_r3.3', '3_11_n20', '4_1_2', '6_15_3', '4_2_3', '3_10', '3_11_n5', '3_11_n17', '3_13_r2.5', '7_2', '8_2_4', '5_14', '6_16', '3_24_n5', '8_1_1', '3_24_n50', '8_13_1', '3_13_r5', '5_8', '3_33', '3_18', '3_4_1', '5_15_2_2', '1_27', '1_18', '1_6', '3_24_n20', '3_29', '7_12', '4_1_4', '8_14', '5_18', '8_3_3', '2_7', '1_30', '6_2_n40', '6_7', '6_8_2', '3_11_n8', '3_24_n60', '5_19_1', '1_33', '8_4_3', '3_19', '5_7_1', '3_25_n70', '4_2_1', '1_20_3', '6_4', '3_12_n5', '4_1_3', '3_24_n40', '1_16', '1_2', '8_1_4', '5_12', '5_4', '6_2_n70', '3_25_n80', '7_15', '5_7_2', '2_5', '6_15_1', '2_3_2', '6_2_n50', '1_26']

cnt=0

import numpy as np

train = False

if train:
    selected_keys = np.random.choice(list(data.keys()), int(len(data.keys()) * 0.75))
else:
    selected_keys = np.random.choice(list(data.keys()), int(len(data.keys()) * 0.25))

for image in selected_keys:
    
    cnt += 1
    
    filename = '/jet/prs/russian_signs/rtsd/rtsd-frames/'+image
    
    encoded_image = tf.gfile.GFile(filename, 'rb').read()

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(int(image_height)),
        'image/width': dataset_util.int64_feature(int(image_width)),
        'image/filename': dataset_util.bytes_feature(filename.encode()),
        'image/source_id': dataset_util.bytes_feature(image.encode()),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature('jpg'.encode()),
        'image/object/bbox/xmin': dataset_util.float_list_feature(data[image]['xmins']),
        'image/object/bbox/xmax': dataset_util.float_list_feature(data[image]['xmaxs']),
        'image/object/bbox/ymin': dataset_util.float_list_feature(data[image]['ymins']),
        'image/object/bbox/ymax': dataset_util.float_list_feature(data[image]['ymaxs']),
        'image/object/class/text': dataset_util.bytes_list_feature([text.encode() for text in data[image]['classes_text']]),
        'image/object/class/label': dataset_util.int64_list_feature([cl.index(text)+1 for text in data[image]['classes_text']]),
        #'image/object/class/label': dataset_util.int64_list_feature(data[image]['classes']),
    }))

    writer.write(tf_example.SerializeToString())

print("wrote", cnt, "images")
    
writer.close()