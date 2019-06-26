/** @file TransposedConvLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef TRANSPOSEDCONVLAYER_H
#define TRANSPOSEDCONVLAYER_H

#include "ConnectedBaseLayer.h"

/**
@class TransposedConvLayer
@brief Conv Class for transpose convolutional Layer elements ( the term "transposed convolution" (tensorflow) is also sometimes denoted "deconvolution" (caffe), "upconvolution" (u-net) or "fractionally strided convolution").
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
                                                                           const size_t aInputSampleNumber) : ConnectedBaseLayer<ConvDataDims, ActivationFunctionType, DataType>(ConvDataDims(aInputDepth, aInputHeight, aInputWidth), ConvDataDims::TransposedConv(aOutputDepth, aInputHeight, aInputWidth, aFilterHeight, aFilterWidth, aPaddingHeight, aPaddingWidth, aStride)),
                                                                                                              mFilterHeight(aFilterHeight),
                                                                                                              mFilterWidth(aFilterWidth),
                                                                                                              mPaddingHeight(aPaddingHeight),
                                                                                                              mPaddingWidth(aPaddingWidth),
                                                                                                              mStride(aStride),
                                                                                                              mInputSampleNumber(aInputSampleNumber),
                                                                                                              mFilterSize(aFilterHeight * aFilterWidth * aOutputDepth)
{
    this->InitParams(mFilterSize, this->mOutputDims.Depth, mFilterSize);
};

template <template <typename> class ActivationFunctionType, typename DataType>
TransposedConvLayer<ActivationFunctionType, DataType>::TransposedConvLayer(const size_t aFilterHeight,
                                                                           const size_t aFilterWidth,
                                                                           const size_t aPaddingHeight,
                                                                           const size_t aPaddingWidth,
                                                                           const size_t aStride,
                                                                           const ConvDataDims aInputDims,
                                                                           const size_t aOutputDepth,
                                                                           const size_t aInputSampleNumber) : ConnectedBaseLayer<ConvDataDims, ActivationFunctionType, DataType>(aInputDims, ConvDataDims::TransposedConv(aOutputDepth, aInputDims.Height, aInputDims.Width, aFilterHeight, aFilterWidth, aPaddingHeight, aPaddingWidth, aStride)),
                                                                                                              mFilterHeight(aFilterHeight),
                                                                                                              mFilterWidth(aFilterWidth),
                                                                                                              mPaddingHeight(aPaddingHeight),
                                                                                                              mPaddingWidth(aPaddingWidth),
                                                                                                              mStride(aStride),
                                                                                                              mInputSampleNumber(aInputSampleNumber),
                                                                                                              mFilterSize(aFilterHeight * aFilterWidth * aOutputDepth)
{
    this->InitParams(mFilterSize, this->mOutputDims.Depth, mFilterSize);
};

template <template <typename> class ActivationFunctionType, typename DataType>
void TransposedConvLayer<ActivationFunctionType, DataType>::ForwardPass()
{
    if (this->mValidInputFlag)
    {
        std::cout << "mWeights" << std::endl;
        std::cout << this->mWeights.rows() << " " << this->mWeights.cols() << std::endl;
        
        // Forward transpose pass
        Eigen::Matrix<DataType, Dynamic, Dynamic> colImage = this->mWeights * this->mInputPtr->transpose();
        this->mOutput = Eigen::Matrix<DataType, Dynamic, Dynamic>::Zero(this->mOutputDims.Height * this->mOutputDims.Width, this->mOutputDims.Depth);
        dlfunctions::col2im(mFilterHeight, mFilterWidth, colImage.data(), this->mOutput.data(), this->mInputDims.Height, this->mInputDims.Width, mFilterSize, this->mOutputDims.Height, this->mOutputDims.Width, this->mOutputDims.Depth, mPaddingHeight, mPaddingWidth, mStride, mInputSampleNumber);
        // std::cout << "mWeights" << std::endl;
        // std::cout << this->mWeights.rows() << " " << this->mWeights.cols() << std::endl;
        // std::cout << "weights" << std::endl;
        // std::cout << this->mWeights  << std::endl;
        // std::cout << "(*mInputPtr)" << std::endl;
        // std::cout << this->mInputPtr->rows() << " " << this->mInputPtr->cols() << std::endl;
        // std::cout << "input" << std::endl;
        // std::cout << *(this->mInputPtr)  << std::endl;
        // std::cout << "col2im sizes:" << std::endl;
        // std::cout << colImage.rows() << " " << colImage.cols() << std::endl;
        // std::cout << "col2im" << std::endl;
        // std::cout << colImage  << std::endl;
        // std::cout << "mOutput" << std::endl;
        // std::cout << this->mOutput.rows() << " " << this->mOutput.cols() << std::endl;
        // std::cout << "output" << std::endl;
        // std::cout << this->mOutput << std::endl;

        this->mOutput = this->mOutput + this->mBiases.replicate(this->mOutputDims.Height * this->mOutputDims.Width, 1);
        this->ActivationFunction.ForwardFunction(this->mOutput);
        // std::cout << "mWeights" << std::endl;
        // std::cout << mWeights.rows() << " " << mWeights.cols() << std::endl;
        // std::cout << "(*mInputPtr)" << std::endl;
        // std::cout << (*mInputPtr).rows() << " " << (*mInputPtr).cols() << std::endl;
        // std::cout << "mOutput" << std::endl;
        // std::cout << mOutput.rows() << " " << mOutput.cols() << std::endl;
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
        // Backprop input from previous layer.
        Eigen::Matrix<DataType, Dynamic, Dynamic> vBackpropInput = *(this->mBackpropInputPtr);
        this->ActivationFunction.BackwardFunction(vBackpropInput);

        Eigen::Matrix<DataType, Dynamic, Dynamic> vBackpropInputTranspose = vBackpropInput.transpose();

        // derivative wrt to bias
        this->mGradientsBiases = vBackpropInputTranspose.rowwise().sum().transpose();

        // derivative wrt filters (dOut/df = In conv Out)
        Eigen::Matrix<DataType, Dynamic, Dynamic> im2ColImage(this->mOutputDims.Height * this->mOutputDims.Width, mFilterSize);
        dlfunctions::im2col(mFilterHeight, mFilterWidth, this->mInputPtr->data(), im2ColImage.data(), this->mOutputDims.Height, this->mOutputDims.Width, mFilterSize, this->mInputDims.Height, this->mInputDims.Width, this->mInputDims.Depth, mPaddingHeight, mPaddingWidth, mStride, mInputSampleNumber);
        this->mGradientsWeights = (vBackpropInputTranspose * im2ColImage).transpose();

        // derivative wrt to input
        dlfunctions::convolution(this->mBackpropOutput, this->mInputDims.Height, this->mInputDims.Width, this->mWeights, mFilterHeight, mFilterWidth, *(this->mInputPtr), this->mInputDims.Height, this->mInputDims.Width, this->mInputDims.Depth, mPaddingHeight, mPaddingWidth, mStride, mInputSampleNumber); 

        // std::cout << "(*mInputPtr)" << std::endl;
        // std::cout << (*mInputPtr).rows() << " " << (*mInputPtr).cols() << std::endl;

        // std::cout << "mBackpropOutput" << std::endl;
        // std::cout << mBackpropOutput.rows() << " " << mBackpropOutput.cols() << std::endl;

        // std::cout << "mWeights" << std::endl;
        // std::cout << mWeights.rows() << " " << mWeights.cols() << std::endl;

        // std::cout << "mGradientsWeights" << std::endl;
        // std::cout << mGradientsWeights.rows() << " " << mGradientsWeights.cols() << std::endl;

        // std::cout << "mGradientsBiases" << std::endl;
        // std::cout << mGradientsBiases.rows() << " " << mGradientsBiases.cols() << std::endl;

        // Update.
        this->UpdateParams();
    }
    else
    {
        throw(std::runtime_error("BackwardPass(): invalid input"));
    };
}

#endif