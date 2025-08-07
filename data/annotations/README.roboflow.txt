
cricket_object_detection - v1 2025-08-06 6:09pm
==============================

This dataset was exported via roboflow.com on August 6, 2025 at 2:57 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 2398 images.
Bat-ball-stumps-batsman-bowler are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 5 versions of each source image:
* 50% probability of horizontal flip
* Random rotation of between -13 and +13 degrees
* Random shear of between -7° to +7° horizontally and -8° to +8° vertically
* Random brigthness adjustment of between -18 and +18 percent
* Random exposure adjustment of between -9 and +9 percent
* Random Gaussian blur of between 0 and 1.9 pixels
* Salt and pepper noise was applied to 1.64 percent of pixels

The following transformations were applied to the bounding boxes of each image:
* Random rotation of between -11 and +11 degrees
* Random exposure adjustment of between -7 and +7 percent


