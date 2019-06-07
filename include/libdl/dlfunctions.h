#ifndef DLFUNCTIONS_H
#define DLFUNCTIONS_H

#include <random>
#include <stdexcept>
#include <string>
#include <iostream>
#include <Eigen/Core>
//#include <spdlog/spdlog.h>

using Eigen::MatrixXd;
// TODO Homogeneous style! size_t, const, hungarian, ...
namespace dlfunctions
{
// First attempt.
// If the input is in format: images stacked horizontally, however does not support padding
template <int FilterHeight, int FilterWidth>
void im2col(MatrixXd *aOutput, const MatrixXd *aInput, const size_t aStride, const size_t aNumChannels, const size_t aImHeight, const size_t aImWidth)
{
    // TODO Check dims right
    size_t limitRow = aImHeight - FilterHeight + 1;
    size_t limitCol = aImWidth - FilterWidth + 1;
    size_t rowIndex = 0;
    for (size_t row = 0; row < limitRow; row = row + aStride)
    {
        for (size_t col = 0; col < limitCol; col = col + aStride)
        {
            for (size_t chan = 0; chan < aNumChannels; chan++)
            {
                size_t coloffset = chan * aImWidth;
                (*aOutput).block<1, FilterHeight * FilterWidth>(rowIndex, 0 + coloffset) = (*aInput).block<FilterHeight, FilterWidth>(row, col + coloffset).transpose().reshaped();
            }
            rowIndex++;
        }
    }
}
// Inner loops adapted from Caffe https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
// If the input is in format vectorized images stacked horizontally, supports padding and multiple inputs
// TODO Homogeneous coding style.
void im2col(const int FilterHeight, const int FilterWidth, const double *img, double *col, size_t aOutHeight, size_t aOutWidth, size_t aOutFields, int height, int width, int channels,
            int pad_w, int pad_h, int aStride, size_t aNumSamples)
{
    //std::cout << FilterHeight << FilterWidth << aOutHeight << aOutWidth << aOutFields << height << width << channels<< "padw " << pad_w << "padh " << pad_h << "stride " << aStride << "samp " << aNumSamples << std::endl;
    int imOffset = aOutHeight * aOutWidth * aOutFields;
    for (size_t vSample = 0; vSample < aNumSamples; vSample++) // TODO This wrong
    {
        for (int c = 0; c < aOutFields; ++c)
        {
            int w_offset = c % FilterWidth;
            int h_offset = (c / FilterWidth) % FilterHeight;
            int c_im = c / (FilterHeight * FilterWidth);
            for (int h = 0; h < aOutHeight; ++h)
            {
                int h_pad = h * aStride - pad_h + h_offset;
                for (int w = 0; w < aOutWidth; ++w)
                {
                    int w_pad = w * aStride - pad_w + w_offset;
                    if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                    {
                        col[(c * aOutHeight + h) * aOutWidth + w] =
                            img[(c_im * height + h_pad) * width + w_pad];
                    }
                    else
                    {
                        col[(c * aOutHeight + h) * aOutWidth + w] = 0;
                    }
                }
            }
        }
    }
}
MatrixXd flip(const MatrixXd &aFilters, const size_t aNumberCuts)
{
    size_t vFilterSize = aFilters.rows();
    size_t vOutputDepth = aFilters.cols();
    size_t v2DFilterSize = vFilterSize / aNumberCuts;
    MatrixXd vToBeReturned = aFilters;

    for (int i = 0; i < aNumberCuts; ++i)
    {
        vToBeReturned.block(v2DFilterSize * i, 0, v2DFilterSize, vOutputDepth).colwise().reverseInPlace();
    }
    return vToBeReturned;
}

MatrixXd flatten(const MatrixXd &aInput, const size_t aNumberCuts)
{
    size_t vInputDim1 = aInput.rows();
    size_t vInputDim2 = aInput.cols();
    size_t vBlockSize = vInputDim1 / aNumberCuts;
    size_t vOutputCols = vInputDim2 * vBlockSize;
    MatrixXd vToBeReturned(aNumberCuts, vOutputCols);

    for (int i = 0; i < aNumberCuts; ++i)
    {
        MatrixXd vSampleBlock = aInput.block(i * vBlockSize, 0, vBlockSize, vInputDim2);
        vSampleBlock.resize(1, vOutputCols);
        vToBeReturned.row(i) = vSampleBlock;
    }
    return vToBeReturned;
}
MatrixXd unflatten(const MatrixXd &aInput, const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth)
{
    size_t vNumberSamples = aInput.rows();
    size_t vUnflatSampleSize = aInputHeight * aInputWidth / vNumberSamples;
    MatrixXd vToBeReturned(vNumberSamples * vUnflatSampleSize, aInputDepth);
    for (int i = 0; i < vNumberSamples; ++i)
    {
        MatrixXd vSampleBlock = aInput.row(i);
        vSampleBlock.resize(vUnflatSampleSize, aInputDepth);
        vToBeReturned.block(i * vUnflatSampleSize, 0, vUnflatSampleSize, aInputDepth) = vSampleBlock;
    }
    return vToBeReturned;
}

void convolution(MatrixXd &aConvolutedOutput, size_t aOutHeight, size_t aOutWidth, const MatrixXd &aFilters, const int aFilterHeight, const int aFilterWidth, const MatrixXd &aInputImage, int height, int width, int channels,
                 int pad_w, int pad_h, int aStride, size_t aNumSamples)
{
    size_t vOutFields = aFilterHeight * aFilterWidth * channels;
    MatrixXd im2ColImage(aOutHeight * aOutWidth, vOutFields);

    dlfunctions::im2col(aFilterHeight, aFilterWidth, aInputImage.data(), im2ColImage.data(), aOutHeight, aOutWidth, vOutFields, height, width, channels, pad_w, pad_h, aStride, aNumSamples);
    std::cout << "im2ColImage" << std::endl;
    std::cout << im2ColImage << std::endl;
    aConvolutedOutput = im2ColImage * aFilters;
}

void fullconvolution(MatrixXd &aConvolutedOutput, size_t aOutHeight, size_t aOutWidth, const MatrixXd &aFilters, const int aFilterHeight, const int aFilterWidth, const MatrixXd &aInputImage, int height, int width, int channels,
                     int aStride, size_t aNumSamples)
{
    size_t vPadHeight = aFilterHeight - 1;
    size_t vPadWidth = aFilterWidth - 1;
    dlfunctions::convolution(aConvolutedOutput, aOutHeight, aOutWidth, aFilters, aFilterHeight, aFilterWidth, aInputImage, height, width, channels, vPadHeight, vPadWidth, aStride, aNumSamples);
}
void ReLUActivationFunction(MatrixXd &aOutput)
{
    aOutput = aOutput.cwiseMax(0);
}
} // namespace dlfunctions

#endif