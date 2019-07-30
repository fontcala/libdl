# libdl

Here are the high functionality tests with real Data.


Very basic Python bindings are made for each test. The only purpose of these bindings are:
- Testing real case scenarios using using python capabilities to handle image data.
- illustrating how one could use this library for various kinds of deep learning problems.

The c++ code in AutoEncoderExamples, SegmentationExamples and ClassificationExamples illustrates some different ways of using the library, which may or may not be useful to a user. Please refer to the documentation (target doc_doxygen) for proper guidelines.


**Note:** In all of these examples the loss value has not been normalized, normalization of a loss value (eg: by the number of pixels or samples) is left to the user. Therefore loss values might appear relatively large.







