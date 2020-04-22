import tensorflow.contrib.tensorrt as trt
from object_detection.protos import pipeline_pb2
from object_detection import exporter
from google.protobuf import text_format
import tensorflow as tf
import sys
import os.path
import subprocess

from graph_utils import force_nms_cpu as f_force_nms_cpu
from graph_utils import replace_relu6 as f_replace_relu6
from graph_utils import remove_assert as f_remove_assert

## Initialize variables.
OUTPUT_DIR = os.getcwd()
PRECISION_MODE_LIST = ["FP32","FP16","INT8"]

CONFIG_PATH = os.path.join(OUTPUT_DIR, 'data', 'pipeline.config')
CKPT_PATH = os.path.join(OUTPUT_DIR, 'data', 'model.ckpt')
GRAPH_PATH = os.path.join(OUTPUT_DIR, 'data', 'trt_graph.pb')

PRECISION_MODE = PRECISION_MODE_LIST[1]
SCORE_THRESHOLD = 0.7

def build_detection_graph(config, checkpoint,
        batch_size=1,
        score_threshold=None,
        input_shape=None,
        output_dir='.generated_model'):
    """Builds a frozen graph for a pre-trained object detection model"""
    
    config_path = config
    checkpoint_path = checkpoint

    # parse config from file
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, 'r') as f:
        text_format.Merge(f.read(), config, allow_unknown_extension=True)

    # override some config parameters
    if config.model.HasField('ssd'):
        #config.model.ssd.feature_extractor.override_base_feature_extractor_hyperparams = True
        if score_threshold is not None:
            config.model.ssd.post_processing.batch_non_max_suppression.score_threshold = score_threshold    
        if input_shape is not None:
            config.model.ssd.image_resizer.fixed_shape_resizer.height = input_shape[0]
            config.model.ssd.image_resizer.fixed_shape_resizer.width = input_shape[1]
    elif config.model.HasField('faster_rcnn'):
        if score_threshold is not None:
            config.model.faster_rcnn.second_stage_post_processing.score_threshold = score_threshold
        if input_shape is not None:
            config.model.faster_rcnn.image_resizer.fixed_shape_resizer.height = input_shape[0]
            config.model.faster_rcnn.image_resizer.fixed_shape_resizer.width = input_shape[1]

    if os.path.isdir(output_dir):
        subprocess.call(['rm', '-rf', output_dir])

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # export inference graph to file (initial)
    with tf.Session(config=tf_config) as tf_sess:
        with tf.Graph().as_default() as tf_graph:
            # print(os.path.abspath(exporter.__file__))
            exporter.export_inference_graph(
                'image_tensor', 
                pipeline_config=config, 
                trained_checkpoint_prefix=checkpoint_path, 
                output_directory=output_dir, 
                input_shape=[batch_size, None, None, 3]
            )

    # read frozen graph from file
    frozen_graph = tf.GraphDef()
    with open(os.path.join(output_dir, 'frozen_inference_graph.pb'), 'rb') as f:
        frozen_graph.ParseFromString(f.read())

    # apply graph modifications
    frozen_graph = f_force_nms_cpu(frozen_graph)
    frozen_graph = f_replace_relu6(frozen_graph)
    frozen_graph = f_remove_assert(frozen_graph)

    output_names = ['detection_boxes', 'detection_classes', 'detection_scores', 'num_detections']

    # remove temporary directory
    subprocess.call(['rm', '-rf', output_dir])

    return frozen_graph, output_names

def main(args):
    print("This is trt converter.")

    frozen_graph, output_names = build_detection_graph(
        config=CONFIG_PATH,
        checkpoint=CKPT_PATH,
        score_threshold=SCORE_THRESHOLD,
        batch_size=1
    )

    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=output_names,
        max_batch_size=1,
        max_workspace_size_bytes=1 << 25,
        precision_mode=PRECISION_MODE,
        minimum_segment_size=50
    )

    with open(GRAPH_PATH, 'wb') as f:
        f.write(trt_graph.SerializeToString())

if __name__=='__main__':
    main(sys.argv)