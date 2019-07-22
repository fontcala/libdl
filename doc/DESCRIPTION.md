# libdl

C++ Deep Learning Library.

---
# 1. Design
In this library, the main building blocks to run Deep Learning methods are implemented. The library is designed such that \c final classes only implement the few mathematically relevant elements of the corresponding block (eg: Convolutions in the Conv Layer), while leaving the more object-related and boilerplate elements for the base classes. This improves clarity, and makes it easier to introduce more classes, and reduces the amount of code of the library.

This library has been designed to be fast, relying in light data structures, limiting shape checks and using im2col (see file dlfunction.h) as the fundamental operation in convolution-like operations.

With the current design Classification, Encoding and Segmentation tasks have been successfully implemented, reusing the same NetworkElement classes and with no significant changes in the application code structure (for further information refer to the python notebook tests).


## Data
No tensors are used and images are never really processed as images. A 3D input data of sizes (x,y,z) is stored in 2D matrices of size (x * y, z) throughout the entire network. By noting that a convolution can be represented as a Matrix multiplication using im2col. The layers where convolution would be used, apply this im2col trick instead. This makes the library much faster.

Feature Data (such as in the xor problem) is also represented as a matrix of (number of Training Samples,number of features).

## Layers
The central element of the library are Layers, which have a common interface (NetworkElement). They receive an input via \c NetworkElement::SetInput() and \c NetworkElement::SetBackpropInput() respectively, and compute an output via \c NetworkElement::ForwardPass() and \c NetworkElement::BackwardPass() respectively, which is accessible via \c NetworkElement::GetOutput() and \c NetworkElement::GetBackpropOutput() respectively. Additionally the method \c NetworkElement::SetData() is intended to interact with external training/test data, typically training or testing data or labels (but also any kind of data that could have an utility in the learning process, eg: priors or pixel-wise loss weight map). 

Every \c Final Layer Class only needs to implement its own constructors and override:
```cpp
void ForwardPass();
void BackwardPass();
```
Further details of these functions is provided in each layer's class documentation.

Layers may have as input and output various kinds of data. Making a common interface (see interface description) for all layers with arbitrary input and output types is not trivial. For this reason, the burden of representig data is moved to \c mInputDims and \c mOutputDims respectively, which together with the template parameters, encode how each layer should use the data, while the data itself is always stored as a matrix  (and pointed to with a matrix pointer) of template type DataType (default double). This additionally allows a very simple and fast access to input data via raw pointer. For further discussion see BaseLayer.

@remarks Why am I not using smart pointers or encapsulation of data?
@remarks - Raw pointers are used because there is no concept of ownership to be implemented (in fact, the data these pointers are meant to point to, is owned by other objects).
Additionally the data in these pointers is accessed many times, and having a wrapper around the actual pointer could be slower.

@remarks Why pointer to data and not pointer to previous and next layers?
@remarks - Like this the layers are easier to interface. It should be up to the user, where he gets the input from,maybe he has a useful function returning some data which he wants to use in between two layers.
Additionally it might be that in the future several inputs from different layers, which means the Layers cannot be modelled as elements of a doubly linked list. 
 


### Computation Layers
Layers with parameters inherit from ConnectedBaseLayer (which provides methods related to parameter setting, initialization and update parameters) and are templated over an Activation Function.  

Parameter initialization is tunned in a He/Xavier style, by setting the value of the variance of the distribution which the random values are going to be drawn from.

For the sake of flexibility, each layer updates their own parameters independently, unlike in other frameworks, this easily allows different update methods (default is Nesterov Momentum) and parameters for each layer. The enum UpdateMethod defines which update methods are available.

Unlike the data, N 3D filters (weights) of sizes (x,y,z) each are stored in 2D matrices of size (x * y * z, N). With this, the amout of reshaping needed before and after im2col is minimal.
For nonconvolutional layers, weights are stored in normal matrix notation of dense networks.

@remarks Why does the constructor of \c ConvLayer and \c TransposedConvLayer take input dimensions, when theycould be deduced later when setting the input?
@remarks - It may appear that there is no need for the input parameters to be known at construction, since they could be  deduced from the input when this is set after construction. However, this reasoning assumes that the previous layer is going to be the only input to the next and that the sizes will always match. But there are cases where this does not hold (eg: In Concatenation Skip layers, two outputs are concatenated, making the Input Depth dependant not only on the previous layer but also on an arbitrary concatenation operation).

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

The easiest way of doing this is by forcing computation layers to accept a template template parameter ActivationFunctionType, that has to implement the following methods inplace modifying an input:
```cpp
void ForwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput);
void BackwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput);
```
See the existing classes for further information.
- class LinearActivation  
- class ReLUActivation
- class SigmoidActivation


No SFINAE or similar techniques on this parameter are applied to ensure nicer compile time errors in case a wrong Activation Function is passed, since this would make classes less readable and anyways C++20 will introduce Concepts.


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
# 3. Coding style:
- Indentation of 4 spaces.
- CamelCase style for variables and functions
    - For variables a lower case prefix indicates the variable use:
        - Function arguments \c a- (eg: \c aInput)
        - class members \c m- (eg: \c mPaddingWidth)
        - constants \c c- (eg: \c cTolerance)
        - all other \c v- (eg: \c vTemp)
    - For functions in the \c dlfunctions namespace (dlfunctions.h), all lowercase is preferred
- Code is documented using Javadoc style Doxygen.
    - commands "remark" and "note" discuss a specific implementation choice.
- Comments use \c //

The standards c++ 11, 14 and 17 are used at own discretion.


---
# 4. Limitations:
- Currently batch processing is only available in Fully Connected Neural Networks, since with matrix notation more than one sample can be processed efficiently. It would introduce some overhead and looping for Convolutional layers. Besides, sgd is often preferred in image data than batch processing for computational reasons, and batch processing can be approximated by setting a high momentum parameter.
- Currently SoftmaxLossLayer is not supported for 3D images.
- Currenlty no shape checks are performed at every layer, only at the final one (Loss Layer).
- Currently it is not easy to set a user defined parameter update method.

# 5. Example Use:
The most basic way to use the library (without \c NetworkHelper) is illustrated in the following example that solves the xor problem:

```cpp
#include <iostream>
#include <libdl/dlfunctions.h>
#include <libdl/dltypes.h>
#include <libdl/FullyConnectedLayer.h>
#include <libdl/L2LossLayer.h>

using Eigen::MatrixXd;
int main()
{
    // inputData:   inputLabels:
    // 0 1          1
    // 1 0          1
    // 0 0          0
    // 1 1          0
    MatrixXd inputData(4, 2);
    inputData << 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0;
    MatrixXd inputLabels(4, 1);
    inputLabels << 1.0, 1.0, 0.0, 0.0;

    // Construct the Layers, making the simplest network able to solve the problem (note only one final node, so Softmax is not suitable.)
    FullyConnectedLayer<SigmoidActivation> firstLayer(2, 2);
    FullyConnectedLayer<SigmoidActivation> secondLayer(2, 1);
    L2LossLayer L2Layer{};

    // Connect the layers.
    secondLayer.SetInput(firstLayer.GetOutput());
    L2Layer.SetInput(secondLayer.GetOutput());
    firstLayer.SetBackpropInput(secondLayer.GetBackpropOutput());
    secondLayer.SetBackpropInput(L2Layer.GetBackpropOutput());

    firstLayer.SetLearningRate(0.05);
    secondLayer.SetLearningRate(0.05);

    // set distribution and Batch Size.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, inputData.rows() - 1);
    const int vBatchSize = 2;
    std::vector<int> vSampleIndices(vBatchSize);
    for (size_t i = 0; i < 12000; i++)
    {
        // Pick N samples form the dataset randomly
        for (size_t k = 0; k < vBatchSize; k++)
        {
            vSampleIndices[k] = dis(gen);
        }
        // Set picked Input samples and corresponding Labels
        MatrixXd inputSample = inputData(vSampleIndices, Eigen::all);
        MatrixXd inputSampleLabel = inputLabels(vSampleIndices, Eigen::all);
        firstLayer.SetData(inputSample);
        L2Layer.SetData(inputSampleLabel);

        // Forward and Backward Passes
        firstLayer.ForwardPass();
        secondLayer.ForwardPass();
        L2Layer.ForwardPass();
        L2Layer.BackwardPass();
        secondLayer.BackwardPass();
        firstLayer.BackwardPass();

        // Test with the whole data
        if (i % 100 == 0)
        {
            firstLayer.SetData(inputData);
            L2Layer.SetData(inputLabels);
            firstLayer.ForwardPass();
            secondLayer.ForwardPass();
            L2Layer.ForwardPass();
            std::cout << "----- Total Unnormalized Loss: ----" << std::endl;
            std::cout << L2Layer.GetLoss() << std::endl;
            std::cout << "----- Computed Output: ----" << std::endl;
            std::cout << *(secondLayer.GetOutput()) << std::endl;
            std::cout << "----- Total Labels: --" << std::endl;
            std::cout << inputLabels << std::endl;
        }
    }

    std::cout << "Rerun a few times (random weight initialization)" << std::endl;
}
```


The following example illustrates a possible way to use Layers, Activation Functions and NetworkHelper to create a class for training and testing on some data:
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
# Authors

* **Adria Font Calvarons**

