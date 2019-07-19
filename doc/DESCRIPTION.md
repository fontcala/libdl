# libdl

Deep Learning Library - Details

## Design
In this library, the main building blocks to run Deep Learning methods are implemented. The library is designed such that \c final classes only implement the few mathematically relevant elements of the corresponding block (eg: Convolutions in the Conv Layer), while leaving the more object-related and boilerplate elements are left for the base classes. This improves clarity, and makes it easier to introduce more classes, and reduces the amount of code of the library.

The central element of the library are Layers, which have a common interface and receive an input via \c SetInput and \c Set BackpropInput respectively, and compute an output via \c ForwardPass and \c BackwardPass respectively, which is accessible via \c GetOutput and \c GetBackpropOutput respectively.



## Customize libdl
libdl is easily customizable, you can add further functionality by appropriately inheriting from the base layers and implementing a very minimal interface.

### Layers: 
Every \c Final Layer Class only needs to implement its own constructors and override:
```
void ForwardPass();
void BackwardPass();
```
Further details of these functions in each layer's documentation.
#### Computation Layers:
Layers with parameters inherit from ConnectedBaseLayer and are templated over an Activation Function.

- class NetworkElement  
    - class BaseLayer  
            - class ConnectedBaseLayer  
                - class FullyConnectedLayer  
                - class ConvLayer  
                - class TransposedConvLayer
                - (your custom Computation Layer, eg: Depth-wise Convolution) 

#### Loss Layers:
- class NetworkElement 
    - class BaseLayer 
        - class LossBaseLayer 
            - class L2LossLayer
            - class SoftmaxLossLayer
            - (your custom Loss Layers, eg: Triplet Loss)

#### Pooling and Transition Layers:
- class NetworkElement 
    - class BaseLayer 
        - class MaxPoolLayer
        - class FlattenLayer
        - (your custom Pooling/Transition, eg: Global Average Pooling)    


### Activation Functions:
Activation Functions are not Layers, but used by Layers successfully as long as they implement:
```
void ForwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput);
void BackwardFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput);
```
See the existing classes for further information.
- class LinearActivation  
- class ReLUActivation
- class SigmoidActivation
- (your custom Activation Function, eg: Tanh)




### Integration Classes:
Additionally this library provides a simple integration class. However, the user may decide to use it or not, since it does not introduce any particular functionality.
- class NetworkHelper


## Authors

* **Adria Font Calvarons**

