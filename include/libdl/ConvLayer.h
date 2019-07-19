/** @file ConvLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef CONVLAYER_H
#define CONVLAYER_H

#include "ConnectedBaseLayer.h"

/**
* @class ConvLayer
* @brief Conv Class for convolutional Layer elements.
*
* @copydetails FullyConnectedLayer
*
* Convolution is abstracted as a matrix multiplication in both forward and backward passes, for speed purposes.
* and this allows GEMM (further read: https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/)
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
              const size_t aInputSampleNumber,
              const UpdateMethod aUpdateMethod = UpdateMethod::NESTEROV);

    ConvLayer(const size_t aFilterHeight,
              const size_t aFilterWidth,
              const size_t aPaddingHeight,
              const size_t aPaddingWidth,
              const size_t aStride,
              const ConvDataDims aInputDims,
              const size_t aOutputDepth,
              const size_t aInputSampleNumber,
              const UpdateMethod aUpdateMethod = UpdateMethod::NESTEROV);

    ConvLayer(const size_t aFilterHeight,
              const size_t aFilterWidth,
              const size_t aPaddingHeight,
              const size_t aPaddingWidth,
              const size_t aStride,
              const ConvDataDims aInputDims,
              const ConvDataDims aOutputDims,
              const size_t aInputSampleNumber,
              const UpdateMethod aUpdateMethod = UpdateMethod::NESTEROV);

    /**
    * ConvLayer::ForwardPass
    * overrides 
    * @copydoc NetworkElement::ForwardPass
    * 
    * The convolution computation is done using im2col (dlfunctions::im2col) for greater speed.
    * This additionally makes the output go through the activation function specified by the \c ActivationFunctionType template parameter.
    * @return Nothing.
    * @throws std::runtime_error runtime error if flag \c mValidInputFlag does not hold.
    * @warning Does not perform any size check before doing the computations.
    */
    void ForwardPass() override;

    /**
    * ConvLayer::BackwardPass
    * overrides 
    * @copydoc NetworkElement::BackwardPass
    * The backpropagation step also involves convolutions so the im2col trick applies nicely as well.
    * @return Nothing.
    * @throws std::runtime_error runtime error if flag \c mValidInputFlag does not hold.
    * @warning Does not perform any size check before doing the computations.
    * @remark The im2col matrix is being computed again in this case. An alternative would be to store the matrix computed during the forward pass.
    * However this raises memory concerns, since im2col generates very large images (much larger than the input if the kernels overlap)
    */
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
                                                       const size_t aInputSampleNumber,
                                                       const UpdateMethod aUpdateMethod) : ConnectedBaseLayer<ConvDataDims, ActivationFunctionType, DataType>(ConvDataDims(aInputDepth, aInputHeight, aInputWidth), ConvDataDims::NormalConv(aOutputDepth, aInputHeight, aInputWidth, aFilterHeight, aFilterWidth, aPaddingHeight, aPaddingWidth, aStride), aUpdateMethod),
                                                                                           mFilterHeight(aFilterHeight),
                                                                                           mFilterWidth(aFilterWidth),
                                                                                           mPaddingHeight(aPaddingHeight),
                                                                                           mPaddingWidth(aPaddingWidth),
                                                                                           mStride(aStride),
                                                                                           mInputSampleNumber(aInputSampleNumber),
                                                                                           mFilterSize(aFilterHeight * aFilterWidth * aInputDepth)
{
    std::cout << "Conv "
              << "In Depth: " << this->mInputDims.Depth << " In Height: " << this->mInputDims.Height << " In Width: " << this->mInputDims.Width << " Out Depth: " << this->mOutputDims.Depth << " Out Height: " << this->mOutputDims.Height << " Out Width: " << this->mOutputDims.Width << std::endl;
    this->InitParams(mFilterSize, this->mOutputDims.Depth, this->mOutputDims.Depth, mFilterSize);
};

template <template <typename> class ActivationFunctionType, typename DataType>
ConvLayer<ActivationFunctionType, DataType>::ConvLayer(const size_t aFilterHeight,
                                                       const size_t aFilterWidth,
                                                       const size_t aPaddingHeight,
                                                       const size_t aPaddingWidth,
                                                       const size_t aStride,
                                                       const ConvDataDims aInputDims,
                                                       const size_t aOutputDepth,
                                                       const size_t aInputSampleNumber,
                                                       const UpdateMethod aUpdateMethod) : ConvLayer(aFilterHeight,
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
                                                       const size_t aInputSampleNumber,
                                                       const UpdateMethod aUpdateMethod) : ConvLayer(aFilterHeight,
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
        this->ActivationFunction.BackwardFunction(vBackpropInput);

        // Derivative wrt to bias
        this->mGradientsBiases = vBackpropInput.colwise().sum();

        // Transpose used below
        Eigen::Matrix<DataType, Dynamic, Dynamic> vBackpropInputTranspose = vBackpropInput.transpose();

        // Derivative wrt filters (im2col computed again, otherwise might be too memory intense to save it and speed really matters in forward not backward.)
        Eigen::Matrix<DataType, Dynamic, Dynamic> im2ColImage(this->mOutputDims.Height * this->mOutputDims.Width, mFilterSize);
        dlfunctions::im2col(mFilterHeight, mFilterWidth, this->mInputPtr->data(), im2ColImage.data(), this->mOutputDims.Height, this->mOutputDims.Width, mFilterSize, this->mInputDims.Height, this->mInputDims.Width, mPaddingHeight, mPaddingWidth, mStride);
        // Compute convolution
        this->mGradientsWeights = (vBackpropInputTranspose * im2ColImage).transpose();

        if (!this->mIsFirstLayerFlag)
        {
            // Derivative wrt to input
            Eigen::Matrix<DataType, Dynamic, Dynamic> colImage = (this->mWeights * vBackpropInputTranspose).transpose();
            this->mBackpropOutput = Eigen::Matrix<DataType, Dynamic, Dynamic>::Zero(this->mInputDims.Height * this->mInputDims.Width, this->mInputDims.Depth);
            dlfunctions::col2im(mFilterHeight, mFilterWidth, colImage.data(), this->mBackpropOutput.data(), this->mOutputDims.Height, this->mOutputDims.Width, mFilterSize, this->mInputDims.Height, this->mInputDims.Width, mPaddingHeight, mPaddingWidth, mStride);
        }

        // Update Parameters
        this->UpdateParams();
    }
    else
    {
        throw(std::runtime_error("BackwardPass(): invalid input"));
    };
}

#endif