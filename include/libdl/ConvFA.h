/** @file ConvAlignmentLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef CONVALIGNMENTLAYER_H
#define CONVALIGNMENTLAYER_H

#include "ConnectedBaseLayer.h"

/**
* @class ConvAlignmentLayer
* @brief Conv Class for convolutional Layer elements.
*
* @copydetails FullyConnectedLayer
*
* Convolution is abstracted as a matrix multiplication in both forward and backward passes (further read: https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/).
*/
template <template <typename> class ActivationFunctionType, typename DataType = double>
class ConvAlignmentLayer final : public ConnectedBaseLayer<ConvDataDims, ActivationFunctionType, DataType>
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
    Eigen::Matrix<DataType, Dynamic, Dynamic> mRandomWeightMatrix;
    // Constructors
    /**
    * Example Use given defined \c vInputDepth \c vInputHeight and \c vInputWidth
    @code
    const size_t vFilterHeight1 = 5;
    const size_t vFilterWidth1 = 5;
    const size_t vPaddingHeight1 = 1;
    const size_t vPaddingWidth1 = 1;
    const size_t vStride1 = 2;
    const size_t vOutputDepth1 = 6;
    const size_t vInputSampleNumber = 1;

    ConvAlignmentLayer firstConvAlignmentLayer(vFilterHeight1,
                             vFilterWidth1,
                             vPaddingHeight1,
                             vPaddingWidth1,
                             vStride1,
                             vInputDepth,
                             vInputHeight,
                             vInputWidth,
                             vOutputDepth1,
                             vInputSampleNumber);
    @endcode
    */
    ConvAlignmentLayer(const size_t aFilterHeight,
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

    ConvAlignmentLayer(const size_t aFilterHeight,
              const size_t aFilterWidth,
              const size_t aPaddingHeight,
              const size_t aPaddingWidth,
              const size_t aStride,
              const ConvDataDims aInputDims,
              const size_t aOutputDepth,
              const size_t aInputSampleNumber,
              const UpdateMethod aUpdateMethod = UpdateMethod::NESTEROV);

    ConvAlignmentLayer(const size_t aFilterHeight,
              const size_t aFilterWidth,
              const size_t aPaddingHeight,
              const size_t aPaddingWidth,
              const size_t aStride,
              const ConvDataDims aInputDims,
              const ConvDataDims aOutputDims,
              const size_t aInputSampleNumber,
              const UpdateMethod aUpdateMethod = UpdateMethod::NESTEROV);

    /**
    * ConvAlignmentLayer::ForwardPass
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
    * ConvAlignmentLayer::BackwardPass
    * overrides 
    * @copydoc NetworkElement::BackwardPass
    * The backpropagation step also involves convolutions so the im2col trick applies nicely as well.
    * Care has been taken to avoid the transpose operation in the col matrix.
    * @return Nothing.
    * @throws std::runtime_error runtime error if flag \c mValidInputFlag does not hold.
    * @warning Does not perform any size check before doing the computations.
    * @remark The im2col matrix is being computed again in this case. An alternative would be to store the matrix computed during the forward pass.
    * However this raises memory concerns, since im2col generates very large images (much larger than the input if the kernels overlap)
    */
    void BackwardPass() override;

    const Eigen::Matrix<DataType, Dynamic, Dynamic>& GetWeights();
    void SetBackpropWeights(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput);
};

template <template <typename> class ActivationFunctionType, typename DataType>
const Eigen::Matrix<DataType, Dynamic, Dynamic>& ConvAlignmentLayer<ActivationFunctionType, DataType>::GetWeights()
{
    return this->mWeights;
};

template <template <typename> class ActivationFunctionType, typename DataType>
void ConvAlignmentLayer<ActivationFunctionType, DataType>::SetBackpropWeights(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput)
{
    if(aInput.cols() != mRandomWeightMatrix.cols())
    {
        throw(std::runtime_error("SetBackpropWeights: invalid backprop input cols"));
    }
    if(aInput.rows() != mRandomWeightMatrix.rows())
    {
        throw(std::runtime_error("SetBackpropWeights: invalid backprop input rows"));
    }
    
    mRandomWeightMatrix = aInput;
};

template <template <typename> class ActivationFunctionType, typename DataType>
ConvAlignmentLayer<ActivationFunctionType, DataType>::ConvAlignmentLayer(const size_t aFilterHeight,
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
    // mRandomWeightMatrix = this->mWeights;
    // Generate Random Weight Matrix
    // mRandomWeightMatrix  = this->mWeights;
    std::random_device rd;
    std::mt19937 vRandom(rd());
    std::normal_distribution<float> vRandDistr(0, sqrt(2 / mFilterSize));
    mRandomWeightMatrix = Eigen::Matrix<DataType, Dynamic, Dynamic>::NullaryExpr(mFilterSize,this->mOutputDims.Depth, [&]() { return vRandDistr(vRandom); });
};

template <template <typename> class ActivationFunctionType, typename DataType>
ConvAlignmentLayer<ActivationFunctionType, DataType>::ConvAlignmentLayer(const size_t aFilterHeight,
                                                       const size_t aFilterWidth,
                                                       const size_t aPaddingHeight,
                                                       const size_t aPaddingWidth,
                                                       const size_t aStride,
                                                       const ConvDataDims aInputDims,
                                                       const size_t aOutputDepth,
                                                       const size_t aInputSampleNumber,
                                                       const UpdateMethod aUpdateMethod) : ConvAlignmentLayer(aFilterHeight,
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
ConvAlignmentLayer<ActivationFunctionType, DataType>::ConvAlignmentLayer(const size_t aFilterHeight,
                                                       const size_t aFilterWidth,
                                                       const size_t aPaddingHeight,
                                                       const size_t aPaddingWidth,
                                                       const size_t aStride,
                                                       const ConvDataDims aInputDims,
                                                       const ConvDataDims aOutputDims,
                                                       const size_t aInputSampleNumber,
                                                       const UpdateMethod aUpdateMethod) : ConvAlignmentLayer(aFilterHeight,
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
void ConvAlignmentLayer<ActivationFunctionType, DataType>::ForwardPass()
{
    if (this->mValidInputFlag)
    {
        // Compute the im2Col Matrix
        Eigen::Matrix<DataType, Dynamic, Dynamic> vIm2ColImage(this->mOutputDims.Height * this->mOutputDims.Width, mFilterSize);
        dlfunctions::im2col(mFilterHeight, mFilterWidth, this->mInputPtr->data(), vIm2ColImage.data(), this->mOutputDims.Height, this->mOutputDims.Width, mFilterSize, this->mInputDims.Height, this->mInputDims.Width, mPaddingHeight, mPaddingWidth, mStride);
        // Compute Convolution
        this->mOutput = vIm2ColImage * this->mWeights;
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
void ConvAlignmentLayer<ActivationFunctionType, DataType>::BackwardPass()
{
    if (this->mValidBackpropInputFlag)
    {
        // Backprop input from next layer.
        Eigen::Matrix<DataType, Dynamic, Dynamic> vBackpropInput = *(this->mBackpropInputPtr);

        // Backpropagation through activation function
        this->ActivationFunction.BackwardFunction(vBackpropInput);

        // Derivative wrt to bias
        this->mGradientsBiases = vBackpropInput.colwise().sum();

        // Derivative wrt filters (im2col computed again, otherwise might be too memory intense to save it and speed really matters in forward not backward.)
        Eigen::Matrix<DataType, Dynamic, Dynamic> vIm2ColImage(this->mOutputDims.Height * this->mOutputDims.Width, mFilterSize);
        dlfunctions::im2col(mFilterHeight, mFilterWidth, this->mInputPtr->data(), vIm2ColImage.data(), this->mOutputDims.Height, this->mOutputDims.Width, mFilterSize, this->mInputDims.Height, this->mInputDims.Width, mPaddingHeight, mPaddingWidth, mStride);
        //  (Compute convolution (avoid transpose of im2Col image!))
        this->mGradientsWeights = (vBackpropInput.transpose() * vIm2ColImage).transpose();

        if (!this->mIsFirstLayerFlag)
        {
            // Derivative wrt to input
            Eigen::Matrix<DataType, Dynamic, Dynamic> vColImage = (vBackpropInput * this->mRandomWeightMatrix.transpose());
            this->mBackpropOutput = Eigen::Matrix<DataType, Dynamic, Dynamic>::Zero(this->mInputDims.Height * this->mInputDims.Width, this->mInputDims.Depth);
            dlfunctions::col2im(mFilterHeight, mFilterWidth, vColImage.data(), this->mBackpropOutput.data(), this->mOutputDims.Height, this->mOutputDims.Width, mFilterSize, this->mInputDims.Height, this->mInputDims.Width, mPaddingHeight, mPaddingWidth, mStride);
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