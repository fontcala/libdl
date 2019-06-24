/** @file MaxPoolLayer.h
 *  @author Adria Font Calvarons
 */
#ifndef MAXPOOLLAYER_H
#define MAXPOOLLAYER_H

#include "BaseLayer.h"
/**
@class MaxPoolLayer
@brief MaxPool Layer, .
 */
template <class DataType = double>
class MaxPoolLayer : public BaseLayer<ConvDataDims, ConvDataDims, DataType>
{
private:
    const size_t mPoolSize;
    const size_t mStride;
    const size_t mInputSampleNumber;

public:
    // Constructors
    MaxPoolLayer(const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth, const size_t aPoolSize, const size_t aStride, const size_t aInputSampleNumber);
    MaxPoolLayer(const ConvDataDims aInputDims, const size_t aPoolSize, const size_t aStride, const size_t aInputSampleNumber);
    void ForwardPass();
    void BackwardPass();
};

template <class DataType>
MaxPoolLayer<DataType>::MaxPoolLayer(const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth, const size_t aPoolSize, const size_t aStride, const size_t aInputSampleNumber) : BaseLayer<ConvDataDims, ConvDataDims, DataType>(ConvDataDims(aInputDepth, aInputHeight, aInputWidth), ConvDataDims(aInputDepth, aInputHeight, aInputWidth, aPoolSize, aStride)),
                                                                                                                                                                                                     mPoolSize(aPoolSize),
                                                                                                                                                                                                     mStride(aStride),
                                                                                                                                                                                                     mInputSampleNumber(aInputSampleNumber){};

template <class DataType>
MaxPoolLayer<DataType>::MaxPoolLayer(const ConvDataDims aInputDims, const size_t aPoolSize, const size_t aStride, const size_t aInputSampleNumber) : BaseLayer<ConvDataDims, ConvDataDims, DataType>(aInputDims, ConvDataDims(aInputDims.Depth, aInputDims.Height, aInputDims.Width, aPoolSize, aStride)),
                                                                                                                                                     mPoolSize(aPoolSize),
                                                                                                                                                     mStride(aStride),
                                                                                                                                                     mInputSampleNumber(aInputSampleNumber){};

template <class DataType>
void MaxPoolLayer<DataType>::ForwardPass()
{
    if (this->mValidInputFlag)
    {
        this->mOutput = Eigen::Matrix<DataType, Dynamic, Dynamic>(this->mOutputDims.Height * this->mOutputDims.Width, this->mOutputDims.Depth);
        for (size_t vChannelIndex = 0; vChannelIndex < this->mInputDims.Depth; vChannelIndex++)
        {
            (this->mOutput).col(vChannelIndex) = dlfunctions::im2pool(mPoolSize, this->mInputPtr->col(vChannelIndex).data(), this->mOutputDims.Height, this->mOutputDims.Width, this->mInputDims.Height, this->mInputDims.Width, mStride, mInputSampleNumber);
        }
        std::cout << "(*mInputPtr)" << std::endl;
        std::cout << this->mInputPtr->rows() << " " << this->mInputPtr->cols() << std::endl;
        std::cout << "mOutput" << std::endl;
        std::cout << (this->mOutput).rows() << " " << (this->mOutput).cols() << std::endl;
    }
    else
    {
        throw(std::runtime_error("ForwardPass(): invalid input"));
    };
};
template <class DataType>
void MaxPoolLayer<DataType>::BackwardPass()
{
    if (this->mValidBackpropInputFlag)
    {
        Eigen::Matrix<DataType, Dynamic, Dynamic> vSingleFeatureCol(this->mOutputDims.Height * this->mOutputDims.Width, mPoolSize * mPoolSize);
        Eigen::Matrix<DataType, Dynamic, Dynamic> vAllFeatureCol(this->mOutputDims.Height * this->mOutputDims.Width, mPoolSize * mPoolSize * this->mInputDims.Depth);
        for (size_t vChannelIndex = 0; vChannelIndex < this->mInputDims.Depth; vChannelIndex++)
        {
            dlfunctions::im2colpool(mPoolSize, this->mInputPtr->col(vChannelIndex).data(), vSingleFeatureCol.data(), this->mOutputDims.Height, this->mOutputDims.Width, this->mInputDims.Height, this->mInputDims.Width, mStride, mInputSampleNumber);
            Eigen::Matrix<bool, Dynamic, Dynamic> someMatBool = (vSingleFeatureCol.colwise() - vSingleFeatureCol.rowwise().maxCoeff()).cwiseAbs().array() < std::numeric_limits<double>::epsilon();
            vAllFeatureCol.block(0, mPoolSize * mPoolSize * vChannelIndex, this->mOutputDims.Height * this->mOutputDims.Width, mPoolSize * mPoolSize) = (this->mBackpropInputPtr->col(vChannelIndex).replicate(1, mPoolSize * mPoolSize)).array() * someMatBool.cast<double>().array();
        }
        this->mBackpropOutput = Eigen::Matrix<DataType, Dynamic, Dynamic>::Zero(this->mInputDims.Height * this->mInputDims.Width, this->mInputDims.Depth);
        dlfunctions::colpool2im(mPoolSize, vAllFeatureCol.data(), (this->mBackpropOutput).data(), this->mOutputDims.Height, this->mOutputDims.Width, mPoolSize * mPoolSize * this->mInputDims.Depth, this->mInputDims.Height, this->mInputDims.Width, mStride, mInputSampleNumber);
        // std::cout << "(*mInputPtr)" << std::endl;
        // std::cout << (*mInputPtr).rows() << " " << (*mInputPtr).cols() << std::endl;
        // std::cout << "mBackpropOutput" << std::endl;
        // std::cout << mBackpropOutput.rows() << " " << mBackpropOutput.cols() << std::endl;
        // std::cout << "(*mBackpropInputPtr)" << std::endl;
        // std::cout << (*mBackpropInputPtr).rows() << " " << (*mBackpropInputPtr).cols() << std::endl;
        // std::cout << "mOutput" << std::endl;
        // std::cout << mOutput.rows() << " " << mOutput.cols() << std::endl;
    }
    else
    {
        throw(std::runtime_error("BackwardPass(): invalid input"));
    };
};
#endif