# Computer Vision. Keypoints matching on satellite photos.

We need to train the model for choosing keypoints, matching keypoints and visualisation keypoints on a satellite photos. 

# Praparing of dataset

Use dataset on Kaggle

https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine

# Model for matching key points in satellite photographs.

We will use the ready-made and debugged LightGlue library:

https://github.com/cvg/LightGlue/blob/main/demo.ipynb

Determining key points occurs in two stages:

1. Determination of features using an extractor.
2. Comparison of features using a matcher.

# Conclusions. 

Even in the smallest photographs we get well-marked key points.

We see that the coincidence lines of the key points do not intersect. This means that the comparison accuracy is high.

We see that the shape of the reservoir, outlined by the dots, is preserved from photograph to photograph.

# Options for improving the accuracy of matching.

You can do additional extraction of key points, for example, using the functions of the OpenCV library.

You can use the keypoint clustering mechanism. The algorithm will find clusters of key points in images and replace them with fewer key points. This will increase the recognition accuracy in conditions when most of the points in the second photo are hidden by snow or clouds.

To increase recognition accuracy, we can use inverse analysis of photographs. We compared photo A with photo B, but did not compare photo B with photo A to speed up the analysis process.  
But we can compare photo A with photo B, invert the comparison and compare photo B with photo A. And then filter out the key points that do not match in both cases. Only the most important key points that are relevant for both photos will remain.
