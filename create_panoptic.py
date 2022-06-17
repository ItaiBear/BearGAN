import tensorflow as tf
import json
import io
import collections
import os
from PIL import ImageOps
from PIL import Image
import numpy as np
#import deeplab2
from deeplab2.data import dataset
from constants import root_project_directory

"""
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor', [
        'dataset_name',  # Dataset name.
        'splits_to_sizes',  # Splits of the dataset into training, val and test.
        'num_classes',   # Number of semantic classes.
        'ignore_label',  # Ignore label value used for semantic segmentation.

        # Fields below are used for panoptic segmentation and will be None for
        # Semantic segmentation datasets.
        # Label divisor only used in panoptic segmentation annotation to infer
        # semantic label and instance id.
        'panoptic_label_divisor',
        # A tuple of classes that contains instance annotations. For example,
        # 'person' class has instance annotations while 'sky' does not.
        'class_has_instances_list',
        # A flag indicating whether the dataset is a video dataset that contains
        # sequence IDs and frame IDs.
        'is_video_dataset',
        # A string specifying the colormap that should be used for
        # visualization. E.g. 'cityscapes'.
        'colormap',
        # A flag indicating whether the dataset contains depth annotation.
        'is_depth_dataset',
        # The ignore label for depth annotations.
        'ignore_depth',
    ]
)

CITYSCAPES_COLORMAP = 'cityscapes'
_CITYSCAPES_PANOPTIC = 'cityscapes_panoptic'
CITYSCAPES_PANOPTIC_INFORMATION = DatasetDescriptor(
    dataset_name=_CITYSCAPES_PANOPTIC,
    splits_to_sizes={'train_fine': 2975,
                     'val_fine': 500,
                     'trainval_fine': 3475,
                     'test_fine': 1525},
    num_classes=19,
    ignore_label=255,
    panoptic_label_divisor=1000,
    class_has_instances_list=tuple(range(11, 19)),
    is_video_dataset=False,
    colormap=CITYSCAPES_COLORMAP,
    is_depth_dataset=False,
    ignore_depth=None,
)
"""
_NUM_SHARDS = 10
_SPLITS_TO_SIZES = dataset.CITYSCAPES_PANOPTIC_INFORMATION.splits_to_sizes
_IGNORE_LABEL = dataset.CITYSCAPES_PANOPTIC_INFORMATION.ignore_label
_CLASS_HAS_INSTANCE_LIST = dataset.CITYSCAPES_PANOPTIC_INFORMATION.class_has_instances_list
_PANOPTIC_LABEL_DIVISOR = dataset.CITYSCAPES_PANOPTIC_INFORMATION.panoptic_label_divisor

# A map from data type to folder name that saves the data.
_FOLDERS_MAP = {
    'image': 'leftImg8bit',
    'label': 'gtFine',
}

# A map from data type to filename postfix.
_POSTFIX_MAP = {
    'image': '_leftImg8bit',
    'label': '_gtFine_labelTrainIds',
}

# A map from data type to data format.
_DATA_FORMAT_MAP = {
    'image': 'png',
    'label': 'png',
}
_PANOPTIC_LABEL_FORMAT = 'raw'


def read_image(image_data):
  """Decodes image from in-memory data.
  Args:
    image_data: Bytes data representing encoded image.
  Returns:
    Decoded PIL.Image object.
  """
  image = Image.open(io.BytesIO(image_data))

  try:
    image = ImageOps.exif_transpose(image)    #keeps the image upright
  except TypeError:
    # capture and ignore this bug:
    # https://github.com/python-pillow/Pillow/issues/3973
    pass

  return image

def _read_segments(cityscapes_root, dataset_split):
  """Reads segments information from json file.
  Args:
    cityscapes_root: String, path to Cityscapes dataset root folder.
    dataset_split: String, dataset split.
  Returns:
    segments_dict: A dictionary that maps `image_id` (common file prefix) to
      a tuple of (panoptic annotation file name, segments). Please refer to
      _generate_panoptic_label() method on the detail structure of `segments`.
  """
  json_filename = os.path.join(
      cityscapes_root, "gtFine", 'cityscapes_panoptic_%s_trainId.json' % dataset_split)
  with tf.io.gfile.GFile(json_filename) as f:
    panoptic_dataset = json.load(f)

  segments_dict = {}
  for annotation in panoptic_dataset['annotations']:
    image_id = annotation['image_id']
    if image_id in segments_dict:
      raise ValueError('Image ID %s already exists' % image_id)
    annotation_file_name = annotation['file_name']
    segments = annotation['segments_info']

    segments_dict[image_id] = (annotation_file_name, segments)
  return segments_dict

def _split_image_path(image_path):
  """Helper method to extract split paths from input image path.
  Args:
    image_path: String, path to the image file.
  Returns:
    A tuple of (cityscape root, dataset split, cityname and shared filename
      prefix).
  """
  image_path = os.path.normpath(image_path)
  path_list = image_path.split(os.sep)
  image_folder, dataset_split, city_name, file_name = path_list[-4:]
  if image_folder != _FOLDERS_MAP['image']:
    raise ValueError('Expects image path %s containing image folder.'
                     % image_path)

  pattern = '%s.%s' % (_POSTFIX_MAP['image'], _DATA_FORMAT_MAP['image'])
  if not file_name.endswith(pattern):
    raise ValueError('Image file name %s should end with %s' %
                     (file_name, pattern))

  file_prefix = file_name[:-len(pattern)]
  return os.sep.join(path_list[:-4]), dataset_split, city_name, file_prefix


def _generate_panoptic_label(panoptic_annotation_file, segments, treat_crowd_as_ignore):
  """Creates panoptic label map from annotations.
  Args:
    panoptic_annotation_file: String, path to panoptic annotation (populated
      with `trainId`).
    segments: A list of dictionaries containing information of every segment.
      Read from panoptic_${DATASET_SPLIT}_trainId.json. This method consumes
      the following fields in each dictionary:
        - id: panoptic id
        - category_id: semantic class id
        - area: pixel area of this segment
        - iscrowd: if this segment is crowd region
  Returns:
    A 2D numpy int32 array with the same height / width with panoptic
    annotation. Each pixel value represents its panoptic ID. Please refer to
    ../g3doc/setup/cityscapes.md for more details about how panoptic ID is
    assigned.
  """
  with tf.io.gfile.GFile(panoptic_annotation_file, 'rb') as f:
    panoptic_label = read_image(f.read())

  if panoptic_label.mode != 'RGB':
    raise ValueError('Expect RGB image for panoptic label, gets %s' %
                     panoptic_label.mode)

  panoptic_label = np.array(panoptic_label, dtype=np.int32)
  # Cityscapes panoptic map is created by:
  #   color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
  panoptic_label = np.dot(panoptic_label, [1, 256, 256 * 256])

  semantic_label = np.ones_like(panoptic_label) * _IGNORE_LABEL
  instance_label = np.zeros_like(panoptic_label)
  # Running count of instances per semantic category.
  instance_count = collections.defaultdict(int)
  for segment in segments:
    selected_pixels = panoptic_label == segment['id']
    pixel_area = np.sum(selected_pixels)
    if pixel_area != segment['area']:
      raise ValueError('Expect %d pixels for segment %s, gets %d.' %
                       (segment['area'], segment, pixel_area))

    category_id = segment['category_id']
    semantic_label[selected_pixels] = category_id

    if category_id in _CLASS_HAS_INSTANCE_LIST:
      if segment['iscrowd']:
        # Cityscapes crowd pixels will have instance ID of 0.
        if treat_crowd_as_ignore:
          semantic_label[selected_pixels] = _IGNORE_LABEL
        continue
      # Non-crowd pixels will have instance ID starting from 1.
      instance_count[category_id] += 1
      if instance_count[category_id] >= _PANOPTIC_LABEL_DIVISOR:
        raise ValueError('Too many instances for category %d in this image.' %
                         category_id)
      instance_label[selected_pixels] = instance_count[category_id]
    elif segment['iscrowd']:
      raise ValueError('Stuff class should not have `iscrowd` label.')

  panoptic_label = semantic_label * _PANOPTIC_LABEL_DIVISOR + instance_label
  return panoptic_label.astype(np.int32)



def _get_panoptic_annotation(cityscapes_root, dataset_split,
                             annotation_file_name):
  panoptic_folder = 'cityscapes_panoptic_%s_trainId' % dataset_split
  return os.path.join(cityscapes_root, _FOLDERS_MAP['label'], panoptic_folder, annotation_file_name)


def _create_panoptic_label(image_path, segments_dict, treat_crowd_as_ignore=False):
  """Creates labels for panoptic segmentation."""
  cityscapes_root, dataset_split, _, file_prefix = _split_image_path(image_path)

  annotation_file_name, segments = segments_dict[file_prefix]
  #get the path to the cityscapesscript trainID annotated panoptic segmentation
  panoptic_annotation_file = _get_panoptic_annotation(cityscapes_root, dataset_split, annotation_file_name)

  panoptic_label = _generate_panoptic_label(panoptic_annotation_file, segments, treat_crowd_as_ignore)
  return panoptic_label, _PANOPTIC_LABEL_FORMAT