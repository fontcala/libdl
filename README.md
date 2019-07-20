# libdl

Deep Learning Library 

## Deliverables for the DLFS course
For the **MNIST milestone**, you'll need to:

Clone the repository and run
```
make
```
or
```
make pybindings
```
in the build folder (just like in the CI). The important target is pybindings, this generates the python wrapper module.

Then open Jupyter notebook in the folder /python and run the script Classifier_MNIST_Deliverable (this one is simple, big strides and no max pool, but gets good enough results) or some other Classifier notebook. This files run an Example architecture with parameters that seem to work. The code for this architectures can be seen in the file /python/ClassifierExamples.h (lowest level example without network helper to run the code).

**Note:** To convince yourself that my code works and there is a learning actually taking place I suggest doing the following: Set the Learning rate to a higher value (eg: in the order of 0.1) and you will see the network doesn't learn anything and the accuracy is around 0.1, which is basically just the same as random. Reset the original Learning (0.005) and the accuracy should be above 0.9.

For the **Final Project Milestone**, you'll need to do the same steps but open one of the Segmentation notebooks instead. The results are far from  Similarly refer to /python/SegmentationExamples.h for code details.


**Note:** More information regarding the interaction with Python and the python bindings can be found in the Python readme


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
in the build folder. And check the results in the folder build/doc/html or build/doc/latex.

The documentation includes
* A **Main Page**, which gives details about the library design and important information (reccomended read)
* Information of the **Classes** being used as well as relevant functions (with class diagrams).
* Source **Files** with ommited doxygen blocks

### Prerequisites

C++17 compiler.

## Running the tests

For funtionality tests, check the folder /test.

For Higher-level tests on real data check the /python directory and run the corresponding notebook files.

## Examples
The python notebook files mentioned above serve as examples. Additionally in the folder /examples provides some more example files.

## Contributing
Please refer to the documentation.

## submodules

* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) - Math library
* [Test](https://github.com/catchorg/Catch2) - Test library
* [Pybind11](https://github.com/pybind/pybind11) - Python binding library
* [spdlog](https://github.com/gabime/spdlog) - Logging library

## Authors

* **Adria Font Calvarons**

