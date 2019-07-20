# libdl

C++ Deep Learning Library.

---
# 1. Design
In this library, the main building blocks to run Deep Learning methods are implemented. The library is designed such that \c final classes only implement the few mathematically relevant elements of the corresponding block (eg: Convolutions in the Conv Layer), while leaving the more object-related and boilerplate elements for the base classes. This improves clarity, and makes it easier to introduce more classes, and reduces the amount of code of the library.

## Layers
The central element of the library are Layers, which have a common interface (NetworkElement). They receive an input via \c NetworkElement::SetInput() and \c NetworkElement::SetBackpropInput() respectively, and compute an output via \c NetworkElement::ForwardPass() and \c NetworkElement::BackwardPass() respectively, which is accessible via \c NetworkElement::GetOutput() and \c NetworkElement::GetBackpropOutput() respectively. Additionally the method \c NetworkElement::SetData() is intended to interact with external training/test data, typically training or testing data or labels (but also any kind of data that could have an utility in the learning process, eg: priors or pixel-wise loss weight map). 

Every \c Final Layer Class only needs to implement its own constructors and override:
```cpp
void ForwardPass();
void BackwardPass();
```
Further details of these functions is provided in each layer's class documentation.

### Computation Layers
Layers with parameters inherit from ConnectedBaseLayer (which provides methods related to parameter setting, initialization and update) and are templated over an Activation Function.  

For the sake of flexibility, each layer updates their own parameters independently, unlike in other frameworks, this easily allows different update methods (default is Nesterov Momentum) and parameters for each layer. The enum UpdateMethod defines which update methods are available.

### Loss Layers
The final layer of a CNN is typically a loss layer. Loss layers inherit from LossBaseLayer and have an additional method  LossBaseLayer::GetLoss() that provides a Loss normalized by LossBaseLayer::mLossNormalizationFactor. 

The method \c LossBaseLayer::SetData() overrides \c BaseLayer::SetData(), such that the input to this method is expected to be a Labels matrix. 

Additionally loss layers fill the member \c BaseLayer::mOutput in a meaningful way given the layer peculiarities (eg: class probabilities) during their forward pass.

### Pooling and Transition Layers
All the Layers which do not belong to any of the groups above are typically used for reshaping, downsampling or both and derive directly from BaseLayer.

### Integration Classes
Additionally this library provides a simple integration class NetworkHelper. However, the user may decide to use it or not, since it does not introduce any particular functionality that cannot be achieved by directly using \c NetworkElement objects.

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
Activation Functions are passed as a template parameter to computation layers.
### Minimal interface to be implemented:

```cpp
void ForwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput);
void BackwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput);
```
See the existing classes for further information.
- class LinearActivation  
- class ReLUActivation
- class SigmoidActivation
- (your custom Activation Function, eg: Tanh)


## Authors

* **Adria Font Calvarons**

