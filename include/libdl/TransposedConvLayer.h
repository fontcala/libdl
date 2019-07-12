/** @file TransposedConvLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef TRANSPOSEDCONVLAYER_H
#define TRANSPOSEDCONVLAYER_H

#include "ConnectedBaseLayer.h"

/**
@class TransposedConvLayer
@brief Transposed Conv Class for transposed convolutional Layer elements.
@note Honestly, still don't understand what transposed or fractionally strided means, this class is just based on the illustrations in https://github.com/vdumoulin/conv_arithmetic and made such that all dimensions are as expected (see testing)
 */
template <template <typename> class ActivationFunctionType, typename DataType = double>
class TransposedConvLayer final : public ConnectedBaseLayer<ConvDataDims, ActivationFunctionType, DataType>
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
                        const size_t aInputSampleNumber,
                        const UpdateMethod aUpdateMethod = UpdateMethod::NESTEROV);

    TransposedConvLayer(const size_t aFilterHeight,
                        const size_t aFilterWidth,
                        const size_t aPaddingHeight,
                        const size_t aPaddingWidth,
                        const size_t aStride,
                        const ConvDataDims aInputDims,
                        const size_t aOutputDepth,
                        const size_t aInputSampleNumber,
                        const UpdateMethod aUpdateMethod = UpdateMethod::NESTEROV);

    TransposedConvLayer(const size_t aFilterHeight,
                        const size_t aFilterWidth,
                        const size_t aPaddingHeight,
                        const size_t aPaddingWidth,
                        const size_t aStride,
                        const ConvDataDims aInputDims,
                        const ConvDataDims aOutputDims,
                        const size_t aInputSampleNumber,
                        const UpdateMethod aUpdateMethod = UpdateMethod::NESTEROV);

    // Layer-specific Forward-Backward passes.
    void ForwardPass() override;
    void BackwardPass() override;
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
                                                                           const size_t aInputSampleNumber,
                                                                           const UpdateMethod aUpdateMethod) : ConnectedBaseLayer<ConvDataDims, ActivationFunctionType, DataType>(ConvDataDims(aInputDepth, aInputHeight, aInputWidth), ConvDataDims::TransposedConv(aOutputDepth, aInputHeight, aInputWidth, aFilterHeight, aFilterWidth, aPaddingHeight, aPaddingWidth, aStride),aUpdateMethod),
                                                                                                              mFilterHeight(aFilterHeight),
                                                                                                              mFilterWidth(aFilterWidth),
                                                                                                              mPaddingHeight(aPaddingHeight),
                                                                                                              mPaddingWidth(aPaddingWidth),
                                                                                                              mStride(aStride),
                                                                                                              mInputSampleNumber(aInputSampleNumber),
                                                                                                              mFilterSize(aFilterHeight * aFilterWidth * aInputDepth),
                                                                                                              mTransposedFilterSize(aFilterHeight * aFilterWidth * aOutputDepth)
{
    std::cout << "Tran " << "In Depth: " << this->mInputDims.Depth << " In Height: " << this->mInputDims.Height << " In Width: " << this->mInputDims.Width << " Out Depth: " << this->mOutputDims.Depth << " Out Height: " << this->mOutputDims.Height << " Out Width: " << this->mOutputDims.Width << std::endl;
    this->InitParams(mTransposedFilterSize,this->mInputDims.Depth,this->mOutputDims.Depth, mTransposedFilterSize);
};

template <template <typename> class ActivationFunctionType, typename DataType>
TransposedConvLayer<ActivationFunctionType, DataType>::TransposedConvLayer(const size_t aFilterHeight,
                                                                           const size_t aFilterWidth,
                                                                           const size_t aPaddingHeight,
                                                                           const size_t aPaddingWidth,
                                                                           const size_t aStride,
                                                                           const ConvDataDims aInputDims,
                                                                           const size_t aOutputDepth,
                                                                           const size_t aInputSampleNumber,
                                                                           const UpdateMethod aUpdateMethod) : TransposedConvLayer(aFilterHeight,
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
TransposedConvLayer<ActivationFunctionType, DataType>::TransposedConvLayer(const size_t aFilterHeight,
                                                                           const size_t aFilterWidth,
                                                                           const size_t aPaddingHeight,
                                                                           const size_t aPaddingWidth,
                                                                           const size_t aStride,
                                                                           const ConvDataDims aInputDims,
                                                                           const ConvDataDims aOutputDims,
                                                                           const size_t aInputSampleNumber,
                                                                           const UpdateMethod aUpdateMethod) : TransposedConvLayer(aFilterHeight,
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
void TransposedConvLayer<ActivationFunctionType, DataType>::ForwardPass()
{
    if (this->mValidInputFlag)
    {
        // Transposed Convolution
        Eigen::Matrix<DataType, Dynamic, Dynamic> colImage = (this->mWeights * this->mInputPtr->transpose()).transpose();
        this->mOutput = Eigen::Matrix<DataType, Dynamic, Dynamic>::Constant(this->mOutputDims.Height * this->mOutputDims.Width, this->mOutputDims.Depth,0.0);
        dlfunctions::col2im(mFilterHeight, mFilterWidth, colImage.data(), this->mOutput.data(), this->mInputDims.Height, this->mInputDims.Width, mTransposedFilterSize, this->mOutputDims.Height, this->mOutputDims.Width, mPaddingHeight, mPaddingWidth, mStride);

        // std::cout << "mTransposedFilterSize" << std::endl;
        // std::cout << mTransposedFilterSize << std::endl;
        // std::cout << "mFilterSize" << std::endl;
        // std::cout << mFilterSize << std::endl;
        // std::cout << "mWeights" << std::endl;
        // std::cout << this->mWeights.rows() << " " << this->mWeights.cols() << std::endl;
        // // // std::cout << "weights" << std::endl;
        // // // std::cout << this->mWeights << std::endl;
        // std::cout << "(*mInputPtr)" << std::endl;
        // std::cout << this->mInputPtr->rows() << " " << this->mInputPtr->cols() << std::endl;
        // // // std::cout << "input" << std::endl;
        // // // std::cout << *(this->mInputPtr) << std::endl;
        // std::cout << "col2im sizes:" << std::endl;
        // std::cout << colImage.rows() << " " << colImage.cols() << std::endl;
        // // // std::cout << "col2im" << std::endl;
        // // // std::cout << colImage  << std::endl;
        // std::cout << "mOutput" << std::endl;
        // std::cout << this->mOutput.rows() << " " << this->mOutput.cols() << std::endl;

        //TODO SOLVE THE PROBLEM WITH BIASES! WTF SIZE SHOULD THEY BE??
        // Add biases
        this->mOutput = this->mOutput + this->mBiases.replicate(this->mOutputDims.Height * this->mOutputDims.Width, 1);

        // Activate
        this->ActivationFunction.ForwardFunction(this->mOutput);

        // TODO Erase all this.
        // std::cout << "fwd tran" << std::endl;   
        // std::cout << "input" << std::endl;
        // std::cout << *(this->mInputPtr) << std::endl;
        // std::cout << "output" << std::endl;
        // std::cout << this->mOutput << std::endl;
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
void TransposedConvLayer<ActivationFunctionType, DataType>::BackwardPass()
{
    if (this->mValidBackpropInputFlag)
    {
        //Backprop input from next layer.
        Eigen::Matrix<DataType, Dynamic, Dynamic> vBackpropInput = *(this->mBackpropInputPtr);

        // Backpropagation through activation functions
        this->ActivationFunction.BackwardFunction(vBackpropInput);
        // --Derivative wrt to bias
        this->mGradientsBiases = vBackpropInput.colwise().sum();
        // --Transpose used below
        Eigen::Matrix<DataType, Dynamic, Dynamic> vBackpropInputTranspose = vBackpropInput.transpose();

        // --Derivative wrt filters (dOut/df = In conv Out) transpose means interchange in and out
        Eigen::Matrix<DataType, Dynamic, Dynamic> im2ColImageFilters(this->mInputDims.Height * this->mInputDims.Width, mTransposedFilterSize);
        dlfunctions::im2col(mFilterHeight, mFilterWidth, vBackpropInput.data(), im2ColImageFilters.data(), this->mInputDims.Height, this->mInputDims.Width, mTransposedFilterSize, this->mOutputDims.Height, this->mOutputDims.Width, mPaddingHeight, mPaddingWidth, mStride);
        // Compute convolution
        this->mGradientsWeights = (this->mInputPtr->transpose() * im2ColImageFilters).transpose();

        // std::cout << "colimageFilters" << std::endl;
        // std::cout << im2ColImageFilters.rows() << " " << im2ColImageFilters.cols() << std::endl;
        // std::cout << this->mGradientsWeights << std::endl;

        // Derivative wrt input
        Eigen::Matrix<DataType, Dynamic, Dynamic> im2ColImageOutput(this->mInputDims.Height * this->mInputDims.Width, mTransposedFilterSize);
        dlfunctions::im2col(mFilterHeight, mFilterWidth, vBackpropInput.data(), im2ColImageOutput.data(), this->mInputDims.Height, this->mInputDims.Width, mTransposedFilterSize, this->mOutputDims.Height, this->mOutputDims.Width, mPaddingHeight, mPaddingWidth, mStride);
        this->mBackpropOutput = im2ColImageOutput * this->mWeights;

        // // TODO erase all this
        // std::cout << "bwd tran" << std::endl;
        // std::cout << "(*mBackpropInputPtr)" << std::endl;
        // std::cout << this->mBackpropInputPtr->rows() << " " << this->mBackpropInputPtr->cols() << std::endl;
        // std::cout << "mBackpropOutput" << std::endl;
        // std::cout << this->mBackpropOutput.rows() << " " << this->mBackpropOutput.cols() << std::endl;
        // std::cout << "mWeights" << std::endl;
        // std::cout << this->mWeights.rows() << " " << this->mWeights.cols() << std::endl;
        // std::cout << "mGradientsWeights" << std::endl;
        // std::cout << this->mGradientsWeights.rows() << " " << this->mGradientsWeights.cols() << std::endl;

        // std::cout << this->mBackpropOutput << std::endl;

        // std::cout << "colimageFilters" << std::endl;
        // std::cout << im2ColImageFilters.rows() << " " << im2ColImageFilters.cols() << std::endl;

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