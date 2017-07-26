# samples
Code samples for deep learning applications.

All these scripts are unacceptable from learning point of view as they train a network on a dataset and test it on the same dataset.
However, the purpose of these scripts is just to provide minimal examples of deep learning codes.

All scripts need some ressources expected to be in *data* and produce output in a *tmp* folder in less than 2 mins (on middle cost gpu). All scripts expect *tmp* folder not to exist.
Low size ressources are provided, others are described below.

## inpainting
The script *inpainting.py* gives an example of inpainting code: a deep network is trained to restore an image from a corrupted version of the image.

One can clearly see that some pixel (typically sky corrupted pixel from image *data/4x.png* and *data/3x.png*) are replaced by an external value in the output (*data/4z.png* and *data/3z.png*).

## downscaled segmentation
The script *segmentation_downscaled.py* gives an example of low resolved segmentation code: a deep network is trained to predic a semantic label to each superpixel of the input image.
In other words, the network produces a downscaled semantic map of the corresponding image.

**The script expects vgg weight (vgg16) are present in the data folder.** Weight can be found at https://github.com/jcjohnson/pytorch-vgg.

One can clearly see the similitude between the predicted mask (*z.jpg*) and the ground truth mask (*y.jpg*).

Why not learning the network from scratch (without https://github.com/jcjohnson/pytorch-vgg weight) ?
Well, such learning from scratch will need **much** more time to produce the same result while requiring a **very** specific learning procedure (typically the learning weight value accross iteration).

## segmentation
The script *segmentation.py* does the same thing *segmentation_downscaled.py* but producing a mask with the image size.
A good GPU is expected to run this script.
