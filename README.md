# WWW: A Unified Framework for Explaining What, Where and Why of Neural Networks by Interpretation of Neuron Concepts

This is the source code for [WWW: A Unified Framework for Explaining What, Where and Why 
of Neural Networks by Interpretation of Neuron Concepts]

## Usage
## Preliminaries
It is tested under Ubuntu Linux 20.04 and Python 3.8 environment, and requries some packages to be installed:
* [PyTorch](https://pytorch.org/)
* [numpy](http://www.numpy.org/)
* [CLIP](https://github.com/openai/CLIP)
* [timm](https://github.com/huggingface/pytorch-image-models)

#### dataset

Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the training data and validation data in
`./datasets/ILSVRC-2012/train` and  `./datasets/ILSVRC-2012/val`, respectively.

#### Pre-trained model
For ImageNet, the model we used in the paper is the pre-trained ResNet-50 and vit is provided by Pytorch and timm. The download process
will start upon running.
For places365, please download `http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar` and place in the `./utils` folder.

## Precompute
WWW need precomputing for calculate Shapley value approximation.
Run `./extract_shap.py`.

For ImageNet with ResNet-50 experiment we placed calcualted Class-wise Shapley value in 
`./utils/RN50_ImageNet_class_shap.pkl`.

## Demo
### Example image selection
Run `./example_selection.py`.

### Concept discovery
Run `./concept_matching.py`.

### Heatmap generation
Run `./image_heatmap.py`.

