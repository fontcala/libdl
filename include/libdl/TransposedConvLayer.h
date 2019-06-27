/** @file TransposedConvLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef TRANSPOSEDCONVLAYER_H
#define TRANSPOSEDCONVLAYER_H

#include "ConnectedBaseLayer.h"

/**
@class TransposedConvLayer
@brief Conv Class for convolutional Layer elements.
 */
template <template <typename> class ActivationFunctionType, typename DataType = double>
class TransposedConvLayer : public ConnectedBaseLayer<ConvDataDims, ActivationFunctionType, DataType>
{
protected:
    // Layer-specific properties.
    const size_t mFilterHeight;
    const size_t mFilterWidth;
    const size_t mPaddingHeight;
    const size_t mPaddingWidth;
    const size_t mStride;

    const size_t mInputSampleNumber;
    const size_t mFilterSize;
    const size_t mTransposedFilterSize;

public:
    // Constructors
    TransposedConvLayer(const size_t aFilterHeight,
                        const size_t aFilterWidth,
                        const size_t aPaddingHeight,
                        const size_t aPaddingWidth,
                        const size_t aStride,
                        const size_t aInputDepth,
                        const size_t aInputHeight,
                        const size_t aInputWidth,
                        const size_t aOutputDepth,
                        const size_t aInputSampleNumber);

    TransposedConvLayer(const size_t aFilterHeight,
                        const size_t aFilterWidth,
                        const size_t aPaddingHeight,
                        const size_t aPaddingWidth,
                        const size_t aStride,
                        const ConvDataDims aInputDims,
                        const size_t aOutputDepth,
                        const size_t aInputSampleNumber);

    // Layer-specific Forward-Backward passes.
    void ForwardPass();
    void BackwardPass();
};
template <template <typename> class ActivationFunctionType, typename DataType>
TransposedConvLayer<ActivationFunctionType, DataType>::TransposedConvLayer(const size_t aFilterHeight,
                                                                           const size_t aFilterWidth,
                                                                           const size_t aPaddingHeight,
                                                                           const size_t aPaddingWidth,
                                                                           const size_t aStride,
                                                                           const size_t aInputDepth,
                                                                           const size_t aInputHeight,
                                                                           const size_t aInputWidth,
                                                                           const size_t aOutputDepth,
                                                                           const size_t aInputSampleNumber) : ConnectedBaseLayer<ConvDataDims, ActivationFunctionType, DataType>(ConvDataDims(aInputDepth, aInputHeight, aInputWidth), ConvDataDims::NormalConv(aOutputDepth, aInputHeight, aInputWidth, aFilterHeight, aFilterWidth, aPaddingHeight, aPaddingWidth, aStride)),
                                                                                                              mFilterHeight(aFilterHeight),
                                                                                                              mFilterWidth(aFilterWidth),
                                                                                                              mPaddingHeight(aPaddingHeight),
                                                                                                              mPaddingWidth(aPaddingWidth),
                                                                                                              mStride(aStride),
                                                                                                              mInputSampleNumber(aInputSampleNumber),
                                                                                                              mFilterSize(aFilterHeight * aFilterWidth * aInputDepth),
                                                                                                              mTransposedFilterSize(aFilterHeight * aFilterWidth * aOutputDepth)
{
    this->InitParams(mTransposedFilterSize, this->mInputDims.Depth, mTransposedFilterSize);
};

template <template <typename> class ActivationFunctionType, typename DataType>
TransposedConvLayer<ActivationFunctionType, DataType>::TransposedConvLayer(const size_t aFilterHeight,
                                                                           const size_t aFilterWidth,
                                                                           const size_t aPaddingHeight,
                                                                           const size_t aPaddingWidth,
                                                                           const size_t aStride,
                                                                           const ConvDataDims aInputDims,
                                                                           const size_t aOutputDepth,
                                                                           const size_t aInputSampleNumber) : TransposedConvLayer(aFilterHeight,
                                                                                                                                  aFilterWidth,
                                                                                                                                  aPaddingHeight,
                                                                                                                                  aPaddingWidth,
                                                                                                                                  aStride,
                                                                                                                                  aInputDims.Depth,
                                                                                                                                  aInputDims.Height,
                                                                                                                                  aInputDims.Width,
                                                                                                                                  aOutputDepth,
                                                                                                                                  aInputSampleNumber){};

template <template <typename> class ActivationFunctionType, typename DataType>
void TransposedConvLayer<ActivationFunctionType, DataType>::ForwardPass()
{
    if (this->mValidInputFlag)
    {
        // Transposed Convolution
        Eigen::Matrix<DataType, Dynamic, Dynamic> colImage = this->mWeights * this->mInputPtr->transpose(); //+ biases??
        this->mOutput = Eigen::Matrix<DataType, Dynamic, Dynamic>::Zero(this->mOutputDims.Height * this->mOutputDims.Width, this->mOutputDims.Depth);
        dlfunctions::col2im(mFilterHeight, mFilterWidth, colImage.data(), this->mOutput.data(), this->mInputDims.Height, this->mInputDims.Width, mTransposedFilterSize, this->mOutputDims.Height, this->mOutputDims.Width, mPaddingHeight, mPaddingWidth, mStride);

        std::cout << "mTransposedFilterSize" << std::endl;
        std::cout << mTransposedFilterSize << std::endl;
        std::cout << "mFilterSize" << std::endl;
        std::cout << mFilterSize << std::endl; 
        std::cout << "mWeights" << std::endl;
        std::cout << this->mWeights.rows() << " " << this->mWeights.cols() << std::endl;
        std::cout << "weights" << std::endl;
        std::cout << this->mWeights << std::endl;
        std::cout << "(*mInputPtr)" << std::endl;
        std::cout << this->mInputPtr->rows() << " " << this->mInputPtr->cols() << std::endl;
        std::cout << "input" << std::endl;
        std::cout << *(this->mInputPtr) << std::endl;
        std::cout << "col2im sizes:" << std::endl;
        std::cout << colImage.rows() << " " << colImage.cols() << std::endl;
        // std::cout << "col2im" << std::endl;
        // std::cout << colImage  << std::endl;
        std::cout << "mOutput" << std::endl;
        std::cout << this->mOutput.rows() << " " << this->mOutput.cols() << std::endl;
        std::cout << "output" << std::endl;
        std::cout << this->mOutput << std::endl;

        //TODO SOLVE THE PROBLEM WITH BIASES! WTF SIZE SHOULD THEY BE??
        // Add biases
        //this->mOutput = this->mOutput + this->mBiases.replicate(this->mOutputDims.Height * this->mOutputDims.Width, 1);
        // Activate
        this->ActivationFunction.ForwardFunction(this->mOutput);

    }
    else
    {
        throw(std::runtime_error("ForwardPass(): invalid input"));
    };
}

template <template <typename> class ActivationFunctionType, typename DataType>
void TransposedConvLayer<ActivationFunctionType, DataType>::BackwardPass()
{
    if (this->mValidBackpropInputFlag)
    {
        // Backprop input from next layer.
        Eigen::Matrix<DataType, Dynamic, Dynamic> vBackpropInput = *(this->mBackpropInputPtr);

        // Backpropagation through activation function
        this->ActivationFunction.BackwardFunction(vBackpropInput);

        // --Derivative wrt to bias ??
        //this->mGradientsBiases = vBackpropInput.colwise().sum();

        // --Transpose used below
        Eigen::Matrix<DataType, Dynamic, Dynamic> vBackpropInputTranspose = vBackpropInput.transpose();

        // --Derivative wrt filters (dOut/df = In conv Out)
        // Compute im2col (im2col computed again, otherwise might be too memory intense to save it and speed really matters in forward not backward.)
        Eigen::Matrix<DataType, Dynamic, Dynamic> im2ColImageFilters(this->mOutputDims.Height * this->mOutputDims.Width, mTransposedFilterSize);
        dlfunctions::im2col(mFilterHeight, mFilterWidth, this->mInputPtr->data(), im2ColImageFilters.data(), this->mOutputDims.Height, this->mOutputDims.Width, mTransposedFilterSize, this->mInputDims.Height, this->mInputDims.Width, mPaddingHeight, mPaddingWidth, mStride);
        // Compute convolution
        this->mGradientsWeights = (vBackpropInputTranspose * im2ColImageFilters).transpose();

        // Derivative wrt input
        Eigen::Matrix<DataType, Dynamic, Dynamic> im2ColImageOutput(this->mInputDims.Height * this->mInputDims.Width, mTransposedFilterSize);
        dlfunctions::im2col(mFilterHeight, mFilterWidth, vBackpropInput.data(), im2ColImageOutput.data(), this->mInputDims.Height, this->mInputDims.Width, mTransposedFilterSize, this->mOutputDims.Height, this->mOutputDims.Width, mPaddingHeight, mPaddingWidth, mStride);
        // Compute Convolution
        this->mBackpropOutput = im2ColImageOutput * this->mWeights;
        // TODO erase all this
        
        std::cout << "(*mBackpropInputPtr)" << std::endl;
        std::cout << (*this->mBackpropInputPtr).rows() << " " << (*this->mBackpropInputPtr).cols() << std::endl;

        std::cout << "mBackpropOutput" << std::endl;
        std::cout << this->mBackpropOutput.rows() << " " << this->mBackpropOutput.cols() << std::endl;
        std::cout << this->mBackpropOutput << std::endl;

        std::cout << "mWeights" << std::endl;
        std::cout << this->mWeights.rows() << " " << this->mWeights.cols() << std::endl;

        std::cout << "colimageFilters" << std::endl;
        std::cout << im2ColImageFilters.rows() << " " << im2ColImageFilters.cols() << std::endl;

        std::cout << "colimageOutput" << std::endl;
        std::cout << im2ColImageOutput.rows() << " " << im2ColImageOutput.cols() << std::endl;

        std::cout << "mGradientsWeights" << std::endl;
        std::cout << this->mGradientsWeights.rows() << " " << this->mGradientsWeights.cols() << std::endl;

        std::cout << "mGradientsBiases" << std::endl;
        std::cout << this->mGradientsBiases.rows() << " " << this->mGradientsBiases.cols() << std::endl;

        // Update.
        this->UpdateParams();
    }
    else
    {
        throw(std::runtime_error("BackwardPass(): invalid input"));
    };
}

#endif