import glob
import multiprocessing as mp
import os
import random
import sys
import tarfile
import zipfile
from collections import defaultdict
from io import StringIO

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

import six.moves.urllib as urllib
from object_detection.utils import ops as utils_ops
from utils import visualization_utils as vis_util
from utils import label_map_util

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

if tf.__version__ < '1.4.0':
    raise ImportError(
        'Please upgrade your tensorflow installation to v1.4.* or later!')


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(
    PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 4)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    # if on Python 2, you might need to cast as a float: float(w)/h
    aspect = w / h

    # compute scaling and pad sizing
    if aspect > 1:  # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(
            int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:  # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(
            int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:  # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    # color image but only one color provided
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)):
        padColor = [padColor] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(
        scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img


def run_inference_for_single_image(image):
    # Get handles to input and output tensors

    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {
        output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(
            tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(
            tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(
            tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                   real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                   real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(
        output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


graph = detection_graph
with graph.as_default():
    with tf.Session() as sess:
        animals_fruits_list = ['animals']
        for animals_fruits in animals_fruits_list:
            zl_path = '/data/zl'
            dir_path = f'{zl_path}/ai_challenger_zsl2018_train_test_a_20180321/zsl_a_{animals_fruits}_test_20180321'

            row = 1
            column = 1
            width = 299
            data = []
            image_list = []
            for filename in glob.glob(f'{dir_path}/*.jpg'):  # assuming gif
                image_list.append(filename)
            TEST_IMAGE_PATHS = image_list
            for i in tqdm(range(len(TEST_IMAGE_PATHS))):
                image_path = TEST_IMAGE_PATHS[i]
                s_img = cv2.imread(image_path)
                b, g, r = cv2.split(s_img)       # get b,g,r
                image_np = cv2.merge([r, g, b])     # switch it to rgb
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                # Actual detection.
                output_dict = run_inference_for_single_image(image_np)
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=8)
                # plt.figure(figsize=IMAGE_SIZE)
                # plt.imshow(image_np)
                if output_dict['detection_scores'][0] > 0:
                    y_min, x_min, y_max, x_max = output_dict['detection_boxes'][0]
                    h, w = image_np.shape[:2]
                    y_1 = int(x_min * w)
                    y_2 = int(x_max * w)
                    x_1 = int(y_min * h)
                    x_2 = int(y_max * h)
                    crop_img = image_np[x_1:x_2, y_1:y_2]
                    resize_pad_img = resizeAndPad(crop_img, (width, width))
                else:
                    resize_pad_img = resizeAndPad(image_np, (width, width))
                data.append(resize_pad_img)

                # plt.subplot(row, column, x + 1)
                # plt.imshow(crop_img)

            data = np.array(data)
            np.save(f'{zl_path}/x_test_{animals_fruits}', data)
