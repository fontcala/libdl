
#ifndef SEGMENTATIONLOSSLAYER_H
#define SEGMENTATIONLOSSLAYER_H

#include "LossBaseLayer.h"
/**
@class SegmentationLossLayer
@brief Segmentation Loss Layer. No particular method, looking for a useful loss for the purpose of segmentation
 */
template <class DataType = double>
class SegmentationLossLayer final : public LossBaseLayer<DataType>
{
protected:
    // Data
    Eigen::Matrix<DataType, Dynamic, Dynamic> mGradientHelper;

public:
    // Constructors
    SegmentationLossLayer();

    void ForwardPass() override;
    void BackwardPass() override;
};

template <class DataType>
SegmentationLossLayer<DataType>::SegmentationLossLayer(){};

template <class DataType>
void SegmentationLossLayer<DataType>::ForwardPass()
{
    if (this->mValidInputFlag)
    {
        //check same number of training samples.
        const size_t vLabelNum = this->mLabels.rows();
        const size_t vOutputNum = this->mInputPtr->rows();
        if (vLabelNum == vOutputNum)
        {
            //Segmentation
            Eigen::Matrix<DataType, Dynamic, Dynamic> exp = (*(this->mInputPtr)).array().exp();
            this->mOutput = exp.array().colwise() / exp.rowwise().sum().array();
            Eigen::Matrix<DataType, Dynamic, Dynamic> logprobs = -this->mOutput.array().log();
            Eigen::Matrix<DataType, Dynamic, Dynamic> filtered = logprobs.cwiseProduct(this->mLabels);

            //mOutput = mGradientHelper;
            // std::cout << "(*mInputPtr)" << std::endl;
            // std::cout << (*mInputPtr).rows() << " " << (*mInputPtr).cols() << std::endl;
            // std::cout << "(*mInputPtr)" << std::endl;
            // std::cout << (*mInputPtr) << std::endl;
            // std::cout << "exp" << std::endl;
            // std::cout << exp << std::endl;
            // std::cout << "probs" << std::endl;
            // std::cout << mGradientHelper << std::endl;
            // std::cout << "filtered" << std::endl;
            // std::cout << filtered << std::endl;
            // Loss divided by number of examples
            this->mLoss = filtered.array().sum() / static_cast<double>(vOutputNum);
        }
        else
        {
            throw(std::runtime_error("ComputeLoss(): dimension mismatch"));
        }
    }
    else
    {
        throw(std::runtime_error("ForwardPass(): invalid input"));
    };
};
template <class DataType>
void SegmentationLossLayer<DataType>::BackwardPass()
{
    this->mBackpropOutput = this->mOutput - this->mLabels;
    // std::cout << "(*mInputPtr)" << std::endl;
    // std::cout << (*mInputPtr).rows() << " " << (*mInputPtr).cols() << std::endl;
    // std::cout << "mBackpropOutput" << std::endl;
    // std::cout << this->mBackpropOutput.rows() << " " << this->mBackpropOutput.cols() << std::endl;
};
#endif