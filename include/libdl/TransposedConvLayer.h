/** @file TransposedConvLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef TRANSPOSEDCONVLAYER_H
#define TRANSPOSEDCONVLAYER_H

#include "ConnectedBaseLayer.h"

/**
* @class TransposedConvLayer
* @brief Transposed Conv Class for transposed convolutional Layer elements.
* 
* @copydetails FullyConnectedLayer
* In this layer the forward and the backward pass are 'exchanged' i.e the output is obtained from the same operation as a conv layer obtains the backpropagation output and vice-versa.
* The weights are therefore matching in size with those of a ConvLayer, but not the biases, since the biases add information to the output.
* 
* Illustrations in the following repository help gain an intuition about the concept of transposed convolution https://github.com/vdumoulin/conv_arithmetic.
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
    /** 
    * TransposedConv::ForwardPass
    * overrides 
    * @copydoc NetworkElement::ForwardPass
    * 
    * Care has been taken to avoid the transpose operation in the col matrix.
    * This additionally makes the output go through the activation function specified by the \c ActivationFunctionType template parameter.
    * @return Nothing.
    * @throws std::runtime_error runtime error if flag \c mValidInputFlag does not hold.
    * @warning Does not perform any size check before doing the computations.
    */
    void ForwardPass() override;
    /**
    * TransposedConv::BackwardPass
    * overrides 
    * @copydoc NetworkElement::BackwardPass
    * 
    * The backpropagation step also involves convolutions so the im2col trick applies nicely as well.
    * @return Nothing.
    * @throws std::runtime_error runtime error if flag \c mValidInputFlag does not hold.
    * @warning Does not perform any size check before doing the computations.
    */
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
                                                                           const UpdateMethod aUpdateMethod) : ConnectedBaseLayer<ConvDataDims, ActivationFunctionType, DataType>(ConvDataDims(aInputDepth, aInputHeight, aInputWidth), ConvDataDims::TransposedConv(aOutputDepth, aInputHeight, aInputWidth, aFilterHeight, aFilterWidth, aPaddingHeight, aPaddingWidth, aStride), aUpdateMethod),
                                                                                                               mFilterHeight(aFilterHeight),
                                                                                                               mFilterWidth(aFilterWidth),
                                                                                                               mPaddingHeight(aPaddingHeight),
                                                                                                               mPaddingWidth(aPaddingWidth),
                                                                                                               mStride(aStride),
                                                                                                               mInputSampleNumber(aInputSampleNumber),
                                                                                                               mTransposedFilterSize(aFilterHeight * aFilterWidth * aOutputDepth)
{
    std::cout << "Tran "
              << "In Depth: " << this->mInputDims.Depth << " In Height: " << this->mInputDims.Height << " In Width: " << this->mInputDims.Width << " Out Depth: " << this->mOutputDims.Depth << " Out Height: " << this->mOutputDims.Height << " Out Width: " << this->mOutputDims.Width << std::endl;
    this->InitParams(mTransposedFilterSize, this->mInputDims.Depth, this->mOutputDims.Depth, mTransposedFilterSize);
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
        // Eigen::Matrix<DataType, Dynamic, Dynamic> colImage = (this->mWeights * this->mInputPtr->transpose()).transpose();
        Eigen::Matrix<DataType, Dynamic, Dynamic> vColImage = (*(this->mInputPtr) * this->mWeights.transpose());
        this->mOutput = Eigen::Matrix<DataType, Dynamic, Dynamic>::Constant(this->mOutputDims.Height * this->mOutputDims.Width, this->mOutputDims.Depth, 0.0);
        dlfunctions::col2im(mFilterHeight, mFilterWidth, vColImage.data(), this->mOutput.data(), this->mInputDims.Height, this->mInputDims.Width, mTransposedFilterSize, this->mOutputDims.Height, this->mOutputDims.Width, mPaddingHeight, mPaddingWidth, mStride);

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
        
        // --Derivative wrt filters (dOut/df = In conv Out) transpose means interchange in and out
        Eigen::Matrix<DataType, Dynamic, Dynamic> vIm2ColImageFilters(this->mInputDims.Height * this->mInputDims.Width, mTransposedFilterSize);
        dlfunctions::im2col(mFilterHeight, mFilterWidth, vBackpropInput.data(), vIm2ColImageFilters.data(), this->mInputDims.Height, this->mInputDims.Width, mTransposedFilterSize, this->mOutputDims.Height, this->mOutputDims.Width, mPaddingHeight, mPaddingWidth, mStride);
        // Compute convolution
        this->mGradientsWeights = (this->mInputPtr->transpose() * vIm2ColImageFilters).transpose();


        if (!this->mIsFirstLayerFlag)
        {
            // Derivative wrt input
            Eigen::Matrix<DataType, Dynamic, Dynamic> vIm2ColImageOutput(this->mInputDims.Height * this->mInputDims.Width, mTransposedFilterSize);
            dlfunctions::im2col(mFilterHeight, mFilterWidth, vBackpropInput.data(), vIm2ColImageOutput.data(), this->mInputDims.Height, this->mInputDims.Width, mTransposedFilterSize, this->mOutputDims.Height, this->mOutputDims.Width, mPaddingHeight, mPaddingWidth, mStride);
            this->mBackpropOutput = vIm2ColImageOutput * this->mWeights;
        }

        // Update.
        this->UpdateParams();
    }
    else
    {
        throw(std::runtime_error("BackwardPass(): invalid input"));
    };
}

#endif