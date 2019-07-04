# libdl

Deep Learning Library 

## Deliverable for the DLFS course
BRANCH:  Master
For the MNIST milestone, you'll need to:

Clone the repository and run
```
make
```
or
```
make pybindings
```
in the build folder (just like in the CI). The important target is pybindings, this generates the python wrapper module.

Then open Jupyter notebook in the folder /python and run the script Classifier_MNIST_Deliverable (this one is simple, big strides and no max pool, but gets good enough results) or Classifier_MNIST_Experimental (this one achieves better results by adding maxpooling and smaller strides, it is also slower). This files run an Example architecture with parameters that seem to work. the code for this architecture can be seen in the file /python/ClassifierExamples.h (lowest level example without network helper to run the code), but looks like this:
```

    // Conv 1
    const size_t vFilterHeight1 = 5;
    const size_t vFilterWidth1 = 5;
    const size_t vPaddingHeight1 = 1;
    const size_t vPaddingWidth1 = 1;
    const size_t vStride1 = 2;
    const size_t vOutputDepth1 = 6;

    ConvLayer firstConvLayer(vFilterHeight1,
                             vFilterWidth1,
                             vPaddingHeight1,
                             vPaddingWidth1,
                             vStride1,
                             mInputDepth,
                             mInputHeight,
                             mInputWidth,
                             vOutputDepth1,
                             vInputSampleNumber);

    // Sigmoid.
    SigmoidActivationLayer firstSigmoidLayer;

    // Conv 2
    const size_t vFilterHeight2 = 3;
    const size_t vFilterWidth2 = 3;
    const size_t vPaddingHeight2 = 1;
    const size_t vPaddingWidth2 = 1;
    const size_t vStride2 = 2;
    const size_t vOutputDepth2 = 8;
    ConvLayer secondConvLayer(vFilterHeight2,
                              vFilterWidth2,
                              vPaddingHeight2,
                              vPaddingWidth2,
                              vStride2,
                              firstConvLayer.GetOutputDims(),
                              vOutputDepth2,
                              vInputSampleNumber);

    // Sigmoid
    SigmoidActivationLayer secondSigmoidLayer;

    // flatten layer
    FlattenLayer flattenLayer(secondConvLayer.GetOutputDims(), vInputSampleNumber);

    // fullyconnectedlayer
    FullyConnectedLayer fcLayer(flattenLayer.GetOutputDims(), mNumCategories);

    // losslayer
    SoftmaxLossLayer lossLayer;
```
**Note:** To convince yourself that my code works and there is a learning actually taking place I suggest doing the following: Set the Learning rate to a higher value (eg: in the order of 0.1) and you will see the network doesn't learn anything and the accuracy is around 0.1, which is basically just the same as random. Reset the original Learning (0.005) and the accuracy should be around 0.9.

**Note:** python script runs without problem with version 3.7.1 (and some of the newest scikit sklearn functionality) and ubuntu. Not sure how to check multiplatfrom behaviour.

**Note:** I am not at all proud of the wrapper that I made here, just trying to write the most little amout of python binding possible.

## Library Structure
* class Network
* class NetworkElement
    * class BaseLayer
            * class ConnectedBaseLayer
                * class FullyConnectedLayer
                * class ConvLayer
                * class TransposedConvLayer
        * class LossMeasureLayer 
            * class L2Layer
            * class SoftmaxLayer
            * ...
        * class MaxPoolLayer
        * class FlattenLayer

* class ReLUFunction
* class SigmoidFunction
* ...

Every Layer Class inherits from BaseLayer and only needs to implement:
```
void ForwardPass()
void BackwardPass()
```
Layers with parameters inherit from ConnectedBaseLayer and are templated over an activation Function. 


## Getting Started

Clone the repository and run
```
make
```
in the build folder.
This generates library, tests, and python bindings.
In order to compile the documentation run
```
make doc_doxygen
```
in the build folder. And check the results in the folder build/doc/html or build/doc/latex

### Prerequisites

C++17 compiler.

## Running the tests

TODO pending big changes

## submodules

* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) - Math library
* [Test](https://github.com/catchorg/Catch2) - Test library
* [Pybind11](https://github.com/pybind/pybind11) - Python binding library
* [spdlog](https://github.com/gabime/spdlog) - Logging library

## Authors

* **Adria Font Calvarons**

