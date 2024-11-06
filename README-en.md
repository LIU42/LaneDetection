# Lane Detection

**A simple and efficient model for lane line detection.**

## Overviews

This project constructs a simple and efficient lane line detection model that tries to trade off an acceptable loss of accuracy for higher computational efficiency.

### Model Structure

The model design for this project is referenced in [this paper](https://arxiv.org/abs/2004.11757). The region in the image where the lane lines may be located (the lower part of the image) is meshed to transform the lane line detection problem into a classification problem for the grids where the lane lines are located in each row. The region where the lane lines are located in the image is divided into 12 rows, each row is divided into 80 grids, the model needs to learn how to correctly classify the location of the lane lines in each row based on the input image, and find out the grids where the lane lines are most likely to be present, and the model is able to detect a total of two lane lines on the left side and two on the right side of the image.

#### Design Points

- **Backbone:** The backbone network adopts a network structure similar to that of the YOLO series CSPDarknet as the feature extractor, which reduces the parameters of the model and the amount of computation while trying to guarantee the accuracy. The width and height of the input image is fixed to 640 x 360, and the output feature map format is (B, 256, 12, 20) when image data in (B, 3, 360, 640) format is input.

- **Neck:** The neck module is responsible for further dimensionality reduction of the feature maps extracted by the backbone network, using point convolution to reduce the number of feature map channels from 256 to 4 to reduce the feature dimensions of the data, and finally spreading to generate a 960-dimensional feature vector representation of the entire image.

- **Head:** The detection header module is responsible for generating the final detection results based on the feature vectors of the image, which are first selected by a fully connected layer of features, adjusted for shape and then expanded by dot convolution to obtain the unique heat output of each lane line in the grid where each row is located. Utilizing dot convolution instead of partial full connectivity can effectively reduce the number of parameters in the model and limit the complexity of the model. The format of the final result is (B, 4, 12, 81), which from left to right is the batch size, the number of lane lines (representing two lane lines on the left and two on the right from left to right, respectively), the number of partitioned rows, and the number of grid classifications (the last dimension represents that there are no lane lines in this row).

### Training Strategies

As a model based on the idea of classification, the output results and labels of the model can be regarded as a grid multiclassification problem where all lane lines dividing rows in a batch of image data are located after spreading them from dimension 0 to dimension 2, respectively, to compute the cross-entropy loss.

The model is trained using the [TuSimple](https://www.kaggle.com/datasets/manideep1108/tusimple) dataset, the original labels are processed to meet the requirements of the model training data (see below for the specific requirements), some samples with flawed labeling are eliminated, and the whole The dataset was shuffled and repartitioned, and finally 4343 images were obtained in the training set, 1800 images in the test set, and 500 images in the validation set. The validation set is randomly selected from the test set to determine the convergence of the model and possible overfitting.

The training process uses multi-stage training, starting with a larger learning rate to quickly learn the main features, and thereafter appropriately tuning down the learning rate for more detailed tuning at each training stage.

### Effect Display

The following is a visualization of some of the image model prediction results in the test set, where the dot annotations are the original lane line point coordinate labels, the box annotations are the model prediction result grids, and the red, green, blue, and yellow annotations represent the vehicle's second from the left, first from the left, first from the right, and second from the right lane line detection results, respectively.

![效果展示](assets/examples.jpg "效果展示")

### Performance Evaluation

Taking the lane line detection results labeled in red in Figure 5 as an example, it can be seen that although the prediction results of the model are not exactly the same as the labels, it can still be considered that the model has made correct predictions in general. Considering this kind of situation, referring to the TopK Acc evaluation index in the multi-classification problem, combined with the characteristics of the lane line detection problem, this project adopts DeltaX Acc as the evaluation index of the detection accuracy of the model, which represents the ratio of the number of samples with the distance between the prediction grid and the real labeled grid less than or equal to X units to the total number of samples.

#### Detection Accuracy Metrics

The current optimal model detection accuracy metrics obtained from training are given below:

| Delta0 Acc | Delta1 Acc | Delta2 Acc |
|:----------:|:----------:|:----------:|
| 0.745      | 0.942      | 0.964      |

#### Computational Performance Metrics

In the training and performance testing of the model, the CPU of the local environment is Intel Core i5-1155G7 @ 2.50Ghz, the GPU of the server environment is NVIDIA Tesla P4, and the inference frame rates are all tested in the PyTorch framework environment, and some computational performance metrics of the optimal model obtained from the current training are given below:

| Params（M） | FLOPs（G） | Local CPU Inference Frame Rate（FPS） | Server GPU Inference Frame Rate（FPS) |
|:---------:|:--------:|:-----------------------------------:|:------------------------------------:|
| 1.75      | 0.78     | 45.54                               | 168.82                               |

## Usage

### Environment Setup

First you need to install the various libraries and toolkits that this project depends on.

```bash
pip install -r requirements.txt
```

### Dataset Preparation

The training dataset format for this project is as follows, divided into training set, validation set and test set, all image files need to be resized to 640 x 360 and put into the corresponding images/ directory.

```bash
datasets/
├── test/
│   ├── images/
│   │   ├── xxx.jpg
│   │   ├── xxx.jpg
│   │   └── ...
│   └── annotations.json
├── train/
│   ├── images/
│   │   ├── xxx.jpg
│   │   ├── xxx.jpg
│   │   └── ...
│   └── annotations.json
└── valid/
    ├── images/
    │   ├── xxx.jpg
    │   ├── xxx.jpg
    │   └── ...
    └── annotations.json
```

Where annotations.json is the lane line position annotation file in the dataset division where it is located, its format and description are as follows.

```json5
{
    /*
     * The names of the keys correspond to the names of all image files in the images/ directory.
     */
    "1.jpg": {
        /*
         * The “samples” is a marker for the x-axis coordinates of the lane line points, fixed to a 4x12 two-dimensional array.
         * The first dimensional representation model is able to detect a total of 4 lane lines.
         * In order, they are the second left, first left, first right, and second right lane lines, respectively, centered on the vehicle.
         * For example, the effect shows red, green, blue and yellow lane lines from left to right in Figure 3.
         * The second dimension represents the values of the x-axis coordinates of the lane lines sampled from top to bottom according to the y-axis coordinates in “anchors”.
         * Marked -1 if the lane line does not exist at that position.
         */
        "samples": [
            [...],
            [...],
            [...],
            [...]
        ],
        /*
         * The “anchors” is a sequence of y-axis sampling coordinates for the lane line area, fixed to an array of length 12.
         * This key is not required in the annotation file, but it is necessary to ensure that the sequence of y-axis sampling coordinates is the same for all images in the entire dataset.
         */
        "anchors": [...]
    },
    ...
}
```

### Model Training

After preparing the dataset, run train.py to start training, the default training configuration file is <u>configs/train.yaml</u>, the default configuration and its meaning are as follows:

```yaml
device: "cuda"        # Device name, consistent with PyTroch's device name
epochs: 200           # Number of training iterations
num-workers: 8        # Number of data loading subprocesses
batch-size: 32        # Batch size

learning-rate: 0.0005        # Learning rate
weight-decay: 0.0001         # Weight decay

use-amp: true                # Whether to enable AMP (Automatic Mixing Precision) or not, enabling it will help reduce memory usage and speed up training.
use-augment: true            # Whether to enable image data augment
dropout: 0.5                 # Dropout probability at the fully connected layer
load-checkpoint: false       # Whether to load checkpoint to continue training, if true then load model weights from load-path, otherwise start training with initialized model weights

load-path: "checkpoints/last-ckpt1.pt"    # Initial model load path
best-path: "checkpoints/best-ckpt1.pt"    # Path to save the optimal model for the current validation set
last-path: "checkpoints/last-ckpt1.pt"    # Last training model save path
```

Finetune can also be done based on the model I trained, and the model weights file is published in the Releases of this project.

### Model Evaluation

Once the model is trained, run eval.py to evaluate it. This will calculate the model's Delta0 Acc, Delta1 Acc and Delta2 Acc detection accuracy metrics on the test set, respectively. The default configuration file is <u>configs/eval.yaml</u>, the default configuration and its meaning are as follows:

```yaml
device: "cuda"                                # Device name, consistent with PyTroch's device name
model-path: "checkpoints/best-ckpt3.pt"       # Load paths for models to be evaluated
batch-size: 32                                # Batch size
num-workers: 8                                # Number of data loading subprocesses
```

## Future Plans

- Observation of Figure 6 in the results display and other plots of model prediction results shows that the model does not perform well enough in predicting lane lines with large curvature and at longer distances. The V2 version of the model is being experimented with to address or mitigate the above issues, and I hope it will yield the desired results.

- A model C++ inference deployment program based on ONNX Runtime is under development.
