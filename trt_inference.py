import tensorflow as tf
import numpy as np
import os
import cv2

from graph_utils import force_nms_cpu as f_force_nms_cpu
from graph_utils import replace_relu6 as f_replace_relu6
from graph_utils import remove_assert as f_remove_assert

def main():
    IMG_PATH = os.path.join(os.getcwd(), 'data', 'image.jpg')
    IMG_RESULT_PATH = os.path.join(os.getcwd(), 'data', 'image_result.jpg')
    GRAPH_PATH = os.path.join(os.getcwd(), 'data', 'trt_graph.pb')

    frozen_graph = tf.GraphDef()
    with open(GRAPH_PATH, 'rb') as f:
        frozen_graph.ParseFromString(f.read())

    ## Apply graph modifications
    frozen_graph = f_force_nms_cpu(frozen_graph)
    frozen_graph = f_replace_relu6(frozen_graph)
    frozen_graph = f_remove_assert(frozen_graph)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    tf_sess = tf.Session(config=tf_config)
    tf.import_graph_def(frozen_graph, name='')

    tf_input = tf_sess.graph.get_tensor_by_name('image_tensor:0')
    tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
    tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
    tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
    tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

    image = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    image_result = np.array(image.copy())
    image_expanded = np.expand_dims(cv2.resize(image, (300,300)), axis=0)

    scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={tf_input: image_expanded})

    boxes = boxes[0] # index by 0 to remove batch dimension
    scores = scores[0]
    classes = classes[0]
    num_detections = int(num_detections[0])

    ## Plot boxes
    for i in range(num_detections):
        box = boxes[i] * np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])
        cv2.rectangle(image_result, (int(box[1]), int(box[0])), (int(box[3]),int(box[2])), (0,0,255), 2, 1)

    cv2.imwrite(IMG_RESULT_PATH, image_result)
    cv2.imshow('Object detector', image_result)
    cv2.waitKey(0)

    tf_sess.close()

if __name__=='__main__':
    main()