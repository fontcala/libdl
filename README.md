# libdl

Deep Learning Library 

## Deliverables for the DLFS course
**See the examples folder**


## IMPORTANT NOTE
<span style="color: red">
There are many TODOs, but solving them now is futile since I am considering to following big changes soon (feedback apreciated):
Currently I have FullyConnectedLayer types which encapsulate  aggregation from previous layers and also the Activation Function and Loss Function. I plan to make it separate in the following kind of structure.
* class NetworkElement
    * class AggregationLayer
        * class FullyConnectedLayer
        * class ConvolutionalLayer
    * class ActivationLayer
        * class ReLULayer
        * class SigmoidLayer
    * class LossMeasureLayer 
        * class L2Layer
* class Network
Then it would be easy to add more Activation Functions and Loss Measures.
</span>

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

TODO

## submodules

* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) - Math library
* [Test](https://github.com/catchorg/Catch2) - Test library
* [Pybind11](https://github.com/pybind/pybind11) - Python binding library
* [spdlog](https://github.com/gabime/spdlog) - Logging library

## Authors

* **Adria Font Calvarons**

