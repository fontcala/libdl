/** @file ConvLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef CONVLAYER_H
#define CONVLAYER_H

#include "ConnectedBaseLayer.h"

/**
@class ConvLayer
@brief Conv Class for convolutional Layer elements.
 */
template <template <typename> class ActivationFunctionType, typename DataType = double>
class ConvLayer final : public ConnectedBaseLayer<ConvDataDims, ActivationFunctionType, DataType>
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
    ConvLayer(const size_t aFilterHeight,
              const size_t aFilterWidth,
              const size_t aPaddingHeight,
              const size_t aPaddingWidth,
              const size_t aStride,
              const size_t aInputDepth,
              const size_t aInputHeight,
              const size_t aInputWidth,
              const size_t aOutputDepth,
              const size_t aInputSampleNumber);

    ConvLayer(const size_t aFilterHeight,
              const size_t aFilterWidth,
              const size_t aPaddingHeight,
              const size_t aPaddingWidth,
              const size_t aStride,
              const ConvDataDims aInputDims,
              const size_t aOutputDepth,
              const size_t aInputSampleNumber);

    ConvLayer(const size_t aFilterHeight,
              const size_t aFilterWidth,
              const size_t aPaddingHeight,
              const size_t aPaddingWidth,
              const size_t aStride,
              const ConvDataDims aInputDims,
              const ConvDataDims aOutputDims,
              const size_t aInputSampleNumber);

    // Layer-specific Forward-Backward passes.
    void ForwardPass() override;
    void BackwardPass() override;
};
template <template <typename> class ActivationFunctionType, typename DataType>
ConvLayer<ActivationFunctionType, DataType>::ConvLayer(const size_t aFilterHeight,
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
                                                                                          mFilterSize(aFilterHeight * aFilterWidth * aInputDepth)
{
    // std::cout << aOutputDepth << " aOutputDepth " << aInputHeight << " aInputHeight " <<  " aInputWidth " << aFilterHeight << " aFilterHeight " << aFilterWidth << " aFilterWidth "  << aPaddingHeight << " aPaddingHeight "  << aPaddingWidth << " aPaddingWidth "  << aStride << " aStride " << std::endl;
    std::cout << "this->mInputDims.Height:" << this->mInputDims.Height << " this->mInputDims.Width:" << this->mInputDims.Width << " this->mOutputDims.Height: " << this->mOutputDims.Height << " this->mOutputDims.Width: " << this->mOutputDims.Width << std::endl;
    this->InitParams(mFilterSize, this->mOutputDims.Depth,this->mOutputDims.Depth, mFilterSize);
};

template <template <typename> class ActivationFunctionType, typename DataType>
ConvLayer<ActivationFunctionType, DataType>::ConvLayer(const size_t aFilterHeight,
                                                       const size_t aFilterWidth,
                                                       const size_t aPaddingHeight,
                                                       const size_t aPaddingWidth,
                                                       const size_t aStride,
                                                       const ConvDataDims aInputDims,
                                                       const size_t aOutputDepth,
                                                       const size_t aInputSampleNumber) : ConvLayer(aFilterHeight,
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
ConvLayer<ActivationFunctionType, DataType>::ConvLayer(const size_t aFilterHeight,
                                                       const size_t aFilterWidth,
                                                       const size_t aPaddingHeight,
                                                       const size_t aPaddingWidth,
                                                       const size_t aStride,
                                                       const ConvDataDims aInputDims,
                                                       const ConvDataDims aOutputDims,
                                                       const size_t aInputSampleNumber) : ConvLayer(aFilterHeight,
                                                                                                    aFilterWidth,
                                                                                                    aPaddingHeight,
                                                                                                    aPaddingWidth,
                                                                                                    aStride,
                                                                                                    aInputDims.Depth,
                                                                                                    aInputDims.Height,
                                                                                                    aInputDims.Width,
                                                                                                    aOutputDims.Depth,
                                                                                                    aInputSampleNumber){};

template <template <typename> class ActivationFunctionType, typename DataType>
void ConvLayer<ActivationFunctionType, DataType>::ForwardPass()
{
    if (this->mValidInputFlag)
    {
        // Compute the im2Col Matrix
        Eigen::Matrix<DataType, Dynamic, Dynamic> im2ColImage(this->mOutputDims.Height * this->mOutputDims.Width, mFilterSize);
        dlfunctions::im2col(mFilterHeight, mFilterWidth, this->mInputPtr->data(), im2ColImage.data(), this->mOutputDims.Height, this->mOutputDims.Width, mFilterSize, this->mInputDims.Height, this->mInputDims.Width, mPaddingHeight, mPaddingWidth, mStride);
        // Compute Convolution
        this->mOutput = im2ColImage * this->mWeights;
        // Add biases
        this->mOutput = this->mOutput + this->mBiases.replicate(this->mOutputDims.Height * this->mOutputDims.Width, 1);
        // Activate
        this->ActivationFunction.ForwardFunction(this->mOutput);

        // TODO Erase all this
        // std::cout << "fwd conv" << std::endl;    
        // std::cout << "mWeights" << std::endl;
        // std::cout << this->mWeights.rows() << " " << this->mWeights.cols() << std::endl;
        // std::cout << "(*mInputPtr)" << std::endl;
        // std::cout << (*this->mInputPtr).rows() << " " << (*this->mInputPtr).cols() << std::endl;
        // std::cout << "mOutput" << std::endl;
        // std::cout << this->mOutput.rows() << " " << this->mOutput.cols() << std::endl;
    }
    else
    {
        throw(std::runtime_error("ForwardPass(): invalid input"));
    };
}

template <template <typename> class ActivationFunctionType, typename DataType>
void ConvLayer<ActivationFunctionType, DataType>::BackwardPass()
{
    if (this->mValidBackpropInputFlag)
    {
        // Backprop input from next layer.
        Eigen::Matrix<DataType, Dynamic, Dynamic> vBackpropInput = *(this->mBackpropInputPtr);

        // Backpropagation through activation function
        // std::cout << "vBackpropInput before" << std::endl;
        // std::cout << vBackpropInput << std::endl;
        this->ActivationFunction.BackwardFunction(vBackpropInput);
        // std::cout << "vBackpropInput after" << std::endl;
        // std::cout << vBackpropInput << std::endl;
        // --Derivative wrt to bias
        this->mGradientsBiases = vBackpropInput.colwise().sum();

        // --Transpose used below
        Eigen::Matrix<DataType, Dynamic, Dynamic> vBackpropInputTranspose = vBackpropInput.transpose();

        // --Derivative wrt filters (dOut/df = In conv Out)
        // Compute im2col (im2col computed again, otherwise might be too memory intense to save it and speed really matters in forward not backward.)
        Eigen::Matrix<DataType, Dynamic, Dynamic> im2ColImage(this->mOutputDims.Height * this->mOutputDims.Width, mFilterSize);
        dlfunctions::im2col(mFilterHeight, mFilterWidth, this->mInputPtr->data(), im2ColImage.data(), this->mOutputDims.Height, this->mOutputDims.Width, mFilterSize, this->mInputDims.Height, this->mInputDims.Width, mPaddingHeight, mPaddingWidth, mStride);
        // Compute convolution
        this->mGradientsWeights = (vBackpropInputTranspose * im2ColImage).transpose();

        // Derivative wrt to input
        Eigen::Matrix<DataType, Dynamic, Dynamic> colImage = (this->mWeights * vBackpropInputTranspose).transpose();
        this->mBackpropOutput = Eigen::Matrix<DataType, Dynamic, Dynamic>::Zero(this->mInputDims.Height * this->mInputDims.Width, this->mInputDims.Depth);
        dlfunctions::col2im(mFilterHeight, mFilterWidth, colImage.data(), this->mBackpropOutput.data(), this->mOutputDims.Height, this->mOutputDims.Width, mFilterSize, this->mInputDims.Height, this->mInputDims.Width, mPaddingHeight, mPaddingWidth, mStride);

        // TODO erase all this
        // std::cout << "bwd conv" << std::endl;
        // std::cout << "(*mBackpropInputPtr)" << std::endl;
        // std::cout << this->mBackpropInputPtr->rows() << " " << this->mBackpropInputPtr->cols() << std::endl;
        // std::cout << "mBackpropOutput" << std::endl;
        // std::cout << this->mBackpropOutput.rows() << " " << this->mBackpropOutput.cols() << std::endl;
        // std::cout << "mWeights" << std::endl;
        // std::cout << this->mWeights.rows() << " " << this->mWeights.cols() << std::endl;
        // std::cout << "mGradientsWeights" << std::endl;
        // std::cout << this->mGradientsWeights.rows() << " " << this->mGradientsWeights.cols() << std::endl;
        // std::cout << "(*mBackpropInputPtr)" << std::endl;
        // std::cout << (*this->mBackpropInputPtr).rows() << " " << (*this->mBackpropInputPtr).cols() << std::endl;

        // std::cout << "vBackpropInputTranspose" << std::endl;
        // std::cout << vBackpropInputTranspose.rows() << " " << vBackpropInputTranspose.cols() << std::endl;

        // std::cout << "vBackpropInputTranspose" << std::endl;
        // std::cout << vBackpropInputTranspose << std::endl;

        // std::cout << "mBackpropOutput" << std::endl;
        // std::cout << this->mBackpropOutput.rows() << " " << this->mBackpropOutput.cols() << std::endl;
        // std::cout << this->mBackpropOutput << std::endl;

        // std::cout << "mWeights" << std::endl;
        // std::cout << this->mWeights.rows() << " " << this->mWeights.cols() << std::endl;

        // std::cout << "colimage" << std::endl;
        // std::cout << colImage.rows() << " " << colImage.cols() << std::endl;

        // std::cout << "mGradientsWeights" << std::endl;
        // std::cout << this->mGradientsWeights.rows() << " " << this->mGradientsWeights.cols() << std::endl;

        // std::cout << "mGradientsBiases" << std::endl;
        // std::cout << this->mGradientsBiases.rows() << " " << this->mGradientsBiases.cols() << std::endl;

        // Update.
        this->UpdateParams();
    }
    else
    {
        throw(std::runtime_error("BackwardPass(): invalid input"));
    };
}

#endif