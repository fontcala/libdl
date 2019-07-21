# libdl

C++ Deep Learning Library.

---
# 1. Design
In this library, the main building blocks to run Deep Learning methods are implemented. The library is designed such that \c final classes only implement the few mathematically relevant elements of the corresponding block (eg: Convolutions in the Conv Layer), while leaving the more object-related and boilerplate elements for the base classes. This improves clarity, and makes it easier to introduce more classes, and reduces the amount of code of the library.

With the current design Classification, Encoding and Segmentation tasks have been successfully implemented, reusing the same NetworkElement classes and with no significant changes in the application code structure (for further information refer to the python notebook tests).

## Data
No tensors are used and images are never really processed as images. A 3D input data of sizes (x,y,z) is stored in 2D matrices of size (x * y, z) throughout the entire network. By noting that a convolution can be represented as a Matrix multiplication using im2col. The layers where convolution would be used, apply this im2col trick instead.

Feature Data (such as in the xor problem) is also represented as a matrix of (number of Training Samples,number of features).

## Layers
The central element of the library are Layers, which have a common interface (NetworkElement). They receive an input via \c NetworkElement::SetInput() and \c NetworkElement::SetBackpropInput() respectively, and compute an output via \c NetworkElement::ForwardPass() and \c NetworkElement::BackwardPass() respectively, which is accessible via \c NetworkElement::GetOutput() and \c NetworkElement::GetBackpropOutput() respectively. Additionally the method \c NetworkElement::SetData() is intended to interact with external training/test data, typically training or testing data or labels (but also any kind of data that could have an utility in the learning process, eg: priors or pixel-wise loss weight map). 

Every \c Final Layer Class only needs to implement its own constructors and override:
```cpp
void ForwardPass();
void BackwardPass();
```
Further details of these functions is provided in each layer's class documentation.

Layers may have as input and output various kinds of data. Making a common interface (see interface description) for all layers with arbitrary input and output types is not trivial. For this reason, the burden of representig data is moved to \c mInputDims and \c mOutputDims respectively, which together with the template parameters, encode how each layer should use the data, while the data itself is always stored as a matrix of template type DataType (default double). This additionally allows a very simple and fast access to input data via raw pointer. For further discussion see BaseLayer.


### Computation Layers
Layers with parameters inherit from ConnectedBaseLayer (which provides methods related to parameter setting, initialization and update) and are templated over an Activation Function.  

For the sake of flexibility, each layer updates their own parameters independently, unlike in other frameworks, this easily allows different update methods (default is Nesterov Momentum) and parameters for each layer. The enum UpdateMethod defines which update methods are available.

Unlike the data, N 3D filters (weights) of sizes (x,y,z) each are stored in 2D matrices of size (x * y * z, N). With this, the amout of reshaping needed before and after im2col is minimal.
For nonconvolutional layers, weights are stored in normal matrix notation of dense networks.

### Loss Layers
The final layer of a CNN is typically a loss layer. Loss layers inherit from LossBaseLayer and have an additional method  LossBaseLayer::GetLoss() that provides a Loss normalized by LossBaseLayer::mLossNormalizationFactor. 

The method \c LossBaseLayer::SetData() overrides \c BaseLayer::SetData(), such that the input to this method is expected to be a Labels matrix. 

Additionally loss layers fill the member \c BaseLayer::mOutput in a meaningful way given the layer peculiarities (eg: class probabilities) during their forward pass.

### Pooling and Transition Layers
All the Layers which do not belong to any of the groups above are typically used for reshaping, downsampling or both and derive directly from BaseLayer.

### Integration Classes
Additionally this library provides a simple integration class NetworkHelper. However, the user may decide to use it or not, since it does not introduce any particular functionality that cannot be achieved by directly using \c NetworkElement objects.

## Activation Function.
Activation Functions have not been considered to be layers in the current design for the following reasons:
- Layers don't need any notion of dimension implemented in BaseLayer.
- Since the computation happening in an Activation Function is strictly element-wise and parameterless, there is no need to store the input or the output.
- I want to enforce the use of an Activation Function after every computation Layer (nonlinearities are a must in a neural network).

The easiest way of doing this is by forcing computation layers to accept a template template parameter ActivationFunctionType, that has to implement the following methods:
```cpp
void ForwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput);
void BackwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput);
```
See the existing classes for further information.
- class LinearActivation  
- class ReLUActivation
- class SigmoidActivation


No SFINAE or similar techniques on this parameter are applied to ensure nicer compile time errors in case a wrong Activation Function is passed, since C++20 will introduce Concepts.


---
# 2. Customize libdl

libdl is easily customizable, you can add further functionality by appropriately inheriting from the base layers and implementing a very minimal interface.

## Customize Layers: 

### Minimal interface to be implemented:

```cpp
void ForwardPass();
void BackwardPass();
```
### Where to inherit from:

#### Inheritance for Computation Layers:
- class NetworkElement  
    - class BaseLayer  
            - class ConnectedBaseLayer  
                - class FullyConnectedLayer  
                - class ConvLayer  
                - class TransposedConvLayer
                - (your custom Computation Layer, eg: Depth-wise Convolution) 

#### Inheritance for Loss Layers:
- class NetworkElement 
    - class BaseLayer 
        - class LossBaseLayer 
            - class L2LossLayer
            - class SoftmaxLossLayer
            - (your custom Loss Layers, eg: Triplet Loss)

#### Inheritance for Pooling and Transition Layers:
- class NetworkElement 
    - class BaseLayer 
        - class MaxPoolLayer
        - class FlattenLayer
        - (your custom Pooling/Transition, eg: Global Average Pooling)    


## Customize Activation Functions:
### Minimal interface to be implemented:

```cpp
void ForwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput);
void BackwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput);
```
With that, you may create your custom Activation Function (eg: Tanh).

---
# 3. Example Use:
The following example illustrates a possible way to use Layers, Activation Functions and NetworkHelper to create a class for training and testing on some data
```cpp
#include <libdl/NetworkHelper.h>
#include <libdl/dlfunctions.h>
#include <libdl/dltypes.h>
#include <libdl/ConvLayer.h>
#include <libdl/TransposedConvLayer.h>
#include <libdl/MaxPoolLayer.h>
#include <libdl/L2LossLayer.h>
using Eigen::MatrixXd;
// Autoencoder 
class Example
{
    const size_t mInputHeight;
    const size_t mInputWidth;
    ConvLayer<ReLUActivation> conv1;
    MaxPoolLayer<> maxp1;
    ConvLayer<ReLUActivation> conv2;
    MaxPoolLayer<> maxp2;
    TransposedConvLayer<ReLUActivation> tran3;
    TransposedConvLayer<SigmoidActivation> tran4;
    L2LossLayer<> loss;
    NetworkHelper<> net;

public:
    Example(const size_t aInputHeight,
            const size_t aInputWidth,
            const size_t aInputDepth,
            const size_t aFirstDepth,
            const size_t aSecondDepth) : mInputHeight(aInputHeight),
                                         mInputWidth(aInputWidth),
                                         conv1{3, 3, 1, 1, 1, aInputDepth, aInputHeight, aInputWidth, aFirstDepth, 1, UpdateMethod::ADAM},
                                         maxp1{conv1.GetOutputDims(), 2, 2, 1},
                                         conv2{3, 3, 1, 1, 1, maxp1.GetOutputDims(), aSecondDepth, 1, UpdateMethod::ADAM},
                                         maxp2{conv2.GetOutputDims(), 2, 2, 1},
                                         tran3{2, 2, 0, 0, 2, maxp2.GetOutputDims(), aFirstDepth, 1, UpdateMethod::ADAM},
                                         tran4{2, 2, 0, 0, 2, tran3.GetOutputDims(), aInputDepth, 1, UpdateMethod::ADAM},
                                         loss{10},
                                         net{{&conv1,
                                              &maxp1,
                                              &conv2,
                                              &maxp2,
                                              &tran3,
                                              &tran4,
                                              &loss}}
    {
    }
    void Train(const MatrixXd &aInput, const double aLearningRate, const size_t aEpochNum)
    {
        // This example assumes a user wanting the same learning rate for all layers
        conv1.SetLearningRate(aLearningRate);
        conv2.SetLearningRate(aLearningRate);
        tran3.SetLearningRate(aLearningRate);
        tran4.SetLearningRate(aLearningRate);

        // We need to randomly pick sample indices from the input in each epoch
        std::random_device rd;
        std::mt19937 g(rd());
        const size_t vTotalTrainSamples = aInput.cols();
        std::vector<size_t> vIndexTrainVector(vTotalTrainSamples);
        std::iota(std::begin(vIndexTrainVector), std::end(vIndexTrainVector), 0);
        for (size_t vEpoch = 0; vEpoch < aEpochNum; vEpoch++)
        {
            double vLoss = 0;
            std::shuffle(vIndexTrainVector.begin(), vIndexTrainVector.end(), g);
            for (const auto &vIndex : vIndexTrainVector)
            {
                // take a sample from the data and feed it to the network (in this autoencoder example the label is also the input)
                const MatrixXd Input = aInput.block(0, vIndex, mInputHeight * mInputWidth, 1);
                net.SetInputData(Input);
                net.SetLabelData(Input);
                net.FullForwardPass();
                net.FullBackwardPass();
                vLoss = vLoss + loss.GetLoss();
            }
            std::cout << vLoss / aInput.cols() << std::endl;
        }
    }

    const MatrixXd Test(MatrixXd &aInput)
    {
        net.SetInputData(aInput);
        return net.FullForwardTestPass();
    }
};
```



---
# 4. Coding style:
- Indentation of 4 spaces.
- CamelCase style for variables and functions
    - For variables a lower case prefix indicates the variable use:
        - Function arguments \c a- (eg: \c aInput)
        - class members \c m- (eg: \c mPaddingWidth)
        - constants \c c- (eg: \c cTolerance)
        - all other \c v- (eg: \c vTemp)
- Code is documented using Javadoc style Doxygen.
    - commands "remark" and "note" discuss a specific implementation choice.
- Comments use \c //

The standards c++ 11, 14 and 17 are used at own discretion.


---
# 5. Limitations:
- Currently batch processing is only available in Fully Connected Neural Networks, since with matrix notation more than one sample can be processed efficiently. It would introduce some overhead and looping for Convolutional layers. Besides, sgd is often preferred in image data than batch processing for computational reasons, and batch processing can be approximated by setting a high momentum parameter.
- Currently SoftmaxLossLayer is not supported for 3D images.
- Currenlty no size checks are performed at every layer, only at the final one (Loss Layer).



---
# Authors

* **Adria Font Calvarons**

