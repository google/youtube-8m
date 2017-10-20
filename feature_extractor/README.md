# YouTube8M Feature Extractor
This directory contains binary and library code that can extract YouTube8M
features from images and videos.
The code requires the Inception TensorFlow model ([tutorial](https://www.tensorflow.org/tutorials/image_recognition)) and our PCA matrix, as
outlined in Section 3.3 of our [paper](https://arxiv.org/abs/1609.08675). The
first time you use our code, it will **automatically** download the inception
model (75 Megabytes, tensorflow [GraphDef proto](https://www.tensorflow.org/api_docs/python/tf/GraphDef),
[download link](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz))
and the PCA matrix (25 Megabytes, Numpy arrays,
[download link](http://data.yt8m.org/yt8m_pca.tgz)).

## Usage

There are two ways to use this code:

 1. Binary `extract_tfrecords_main.py` processes a CSV file of videos (and their
    labels) and outputs `tfrecord` file. Files created with this binary match
    the schema of YouTube-8M dataset files, and are therefore are compatible
    with our training starter code. You can also use the file for inference
    using your models that are pre-trained on YouTube-8M.
 1. Library `feature_extractor.py` which can extract features from images.


### Using the Binary to create `tfrecords` from videos

You can use binary `extract_tfrecords_main.py` to create `tfrecord` files.
However, this binary assumes that you have OpenCV properly installed (see end
of subsection). Assume that you have two videos `/path/to/vid1` and
`/path/to/vid2`, respectively, with multi-integer labels of `(52, 3, 10)` and
`(7, 67)`. To create `tfrecord` containing features and labels for those videos,
you must first create a CSV file (e.g. on `/path/to/vid_dataset.csv`) with
contents:

    /path/to/vid1,52;3;10
    /path/to/vid2,7;67

Note that the CSV is comma-separated but the label-field is semi-colon separated
to allow for multiple labels per video.

Then, you can create the `tfrecord` by calling the binary:

    python extract_tfrecords_main.py --input /path/to/vid_dataset.csv \
        --output_tfrecords_file /path/to/output.tfrecord

Now, you can use the output file for training and/or inference using our starter
code.

`extract_tfrecords_main.py` requires OpenCV python bindings to be
installed and linked with ffmpeg. In other words, running this command should
print `True`:

    python -c 'import cv2; print cv2.VideoCapture().open("/path/to/some/video.mp4")'


### Using the library to extract features from images

To extract our features from an image file `cropped_panda.jpg`, you can use
this python code:

```python
from PIL import Image
import numpy

# Instantiate extractor. Slow if called first time on your machine, as it
# needs to download 100 MB.
extractor = YouTube8MFeatureExtractor()

image_file = os.path.join(extractor._model_dir, 'cropped_panda.jpg')

im = numpy.array(Image.open(image_file))
features = extractor.extract_rgb_frame_features(im)
```

The constructor `extractor = YouTube8MFeatureExtractor()` will create a
directory `~/yt8m/`, if it does not exist, and will download and untar the two
model files (inception and PCA matrix). If you prefer, you can point our
extractor to another directory as:

```python
extractor = YouTube8MFeatureExtractor(model_dir="/path/to/yt8m_files")
```

You can also pre-populate your custom `"/path/to/yt8m_files"` by manually
downloading (e.g. using `wget`) the URLs and un-tarring them, for example:

```bash
mkdir -p /path/to/yt8m_files
cd /path/to/yt8m_files

wget http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
wget http://data.yt8m.org/yt8m_pca.tgz

tar zxvf inception-2015-12-05.tgz
tar zxvf yt8m_pca.tgz
```
