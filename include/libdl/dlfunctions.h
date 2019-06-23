#ifndef DLFUNCTIONS_H
#define DLFUNCTIONS_H

#include <random>
#include <stdexcept>
#include <string>
#include <iostream>
#include <Eigen/Core>
//#include <spdlog/spdlog.h>

using Eigen::Dynamic;
namespace dlfunctions
{
// First attempt at im2col. If the input is in format: images stacked horizontally (expects padded input).
template <int FilterHeight, int FilterWidth>
void im2col(Eigen::Matrix<double, Dynamic, Dynamic> *aOutput, const Eigen::Matrix<double, Dynamic, Dynamic> *aInput, const size_t aStride, const size_t aNumChannels, const size_t aImHeight, const size_t aImWidth)
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
// Adapted from Caffe https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
// If the input is in format vectorized images stacked horizontally, can be aeasily made to support padding and multiple inputs
// TODO Homogeneous coding style.
template<class DataType>
void im2col(const int FilterHeight, const int FilterWidth, const DataType *img, DataType *col, size_t aOutHeight, size_t aOutWidth, size_t aOutFields, int height, int width, int channels,
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

// Adapted from Caffe https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
// This is not just a reshaping, it aggregates for locations that have more than one filter in them!
// Also here harder to support multiple images!
template<class DataType>
void col2im(const size_t aFilterHeight, const size_t aFilterWidth, const DataType *aColData, DataType *aImData, size_t aOutHeight, size_t aOutWidth, size_t aOutFields,
            const size_t height, const size_t width, const size_t channels,
            const size_t pad_w, const size_t pad_h,
            const size_t aStride, const size_t aNumSamples)
{
    for (size_t c = 0; c < aOutFields; ++c)
    {
        size_t w_offset = c % aFilterWidth;
        size_t h_offset = (c / aFilterWidth) % aFilterHeight;
        size_t c_im = c / aFilterHeight / aFilterWidth;
        for (size_t h = 0; h < aOutHeight; ++h)
        {
            size_t h_pad = h * aStride - pad_h + h_offset;
            for (size_t w = 0; w < aOutWidth; ++w)
            {
                size_t w_pad = w * aStride - pad_w + w_offset;
                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                    aImData[(c_im * height + h_pad) * width + w_pad] +=
                        aColData[(c * aOutHeight + h) * aOutWidth + w];
            }
        }
    }
}

template<class DataType>
Eigen::Matrix<DataType, Dynamic, Dynamic> flip(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aFilters, const size_t aNumberCuts)
{
    size_t vFilterSize = aFilters.rows();
    size_t vOutputDepth = aFilters.cols();
    size_t v2DFilterSize = vFilterSize / aNumberCuts;
    Eigen::Matrix<DataType, Dynamic, Dynamic> vToBeReturned = aFilters;

    for (int i = 0; i < aNumberCuts; ++i)
    {
        vToBeReturned.block(v2DFilterSize * i, 0, v2DFilterSize, vOutputDepth).colwise().reverseInPlace();
    }
    return vToBeReturned;
}

template<class DataType>
Eigen::Matrix<DataType, Dynamic, Dynamic> flatten(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput, const size_t aNumberCuts)
{
    size_t vInputDim1 = aInput.rows();
    size_t vInputDim2 = aInput.cols();
    size_t vBlockSize = vInputDim1 / aNumberCuts;
    size_t vOutputCols = vInputDim2 * vBlockSize;
    Eigen::Matrix<DataType, Dynamic, Dynamic> vToBeReturned(aNumberCuts, vOutputCols);

    for (int i = 0; i < aNumberCuts; ++i)
    {
        Eigen::Matrix<DataType, Dynamic, Dynamic> vSampleBlock = aInput.block(i * vBlockSize, 0, vBlockSize, vInputDim2);
        vSampleBlock.resize(1, vOutputCols);
        vToBeReturned.row(i) = vSampleBlock;
    }
    return vToBeReturned;
}

template<class DataType>
Eigen::Matrix<DataType, Dynamic, Dynamic> unflatten(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput, const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth)
{
    size_t vNumberSamples = aInput.rows();
    size_t vUnflatSampleSize = aInputHeight * aInputWidth / vNumberSamples;
    Eigen::Matrix<DataType, Dynamic, Dynamic> vToBeReturned(vNumberSamples * vUnflatSampleSize, aInputDepth);
    for (int i = 0; i < vNumberSamples; ++i)
    {
        Eigen::Matrix<DataType, Dynamic, Dynamic> vSampleBlock = aInput.row(i);
        vSampleBlock.resize(vUnflatSampleSize, aInputDepth);
        vToBeReturned.block(i * vUnflatSampleSize, 0, vUnflatSampleSize, aInputDepth) = vSampleBlock;
    }
    return vToBeReturned;
}

template<class DataType>
void convolution(Eigen::Matrix<DataType, Dynamic, Dynamic> &aConvolutedOutput, size_t aOutHeight, size_t aOutWidth, const Eigen::Matrix<DataType, Dynamic, Dynamic> &aFilters, const int aFilterHeight, const int aFilterWidth, const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInputImage, int height, int width, int channels,
                 int pad_w, int pad_h, int aStride, size_t aNumSamples)
{
    size_t vOutFields = aFilterHeight * aFilterWidth * channels;
    Eigen::Matrix<DataType, Dynamic, Dynamic> im2ColImage(aOutHeight * aOutWidth, vOutFields);

    dlfunctions::im2col(aFilterHeight, aFilterWidth, aInputImage.data(), im2ColImage.data(), aOutHeight, aOutWidth, vOutFields, height, width, channels, pad_w, pad_h, aStride, aNumSamples);
    // std::cout << "im2ColImage" << std::endl;
    // std::cout << im2ColImage << std::endl;
    aConvolutedOutput = im2ColImage * aFilters;
}

template<class DataType>
void fullconvolution(Eigen::Matrix<DataType, Dynamic, Dynamic> &aConvolutedOutput, size_t aOutHeight, size_t aOutWidth, const Eigen::Matrix<DataType, Dynamic, Dynamic> &aFilters, const int aFilterHeight, const int aFilterWidth, const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInputImage, int height, int width, int channels,
                     int aStride, size_t aNumSamples)
{
    size_t vPadHeight = aFilterHeight - 1;
    size_t vPadWidth = aFilterWidth - 1;
    dlfunctions::convolution(aConvolutedOutput, aOutHeight, aOutWidth, aFilters, aFilterHeight, aFilterWidth, aInputImage, height, width, channels, vPadHeight, vPadWidth, aStride, aNumSamples);
}
template<class DataType>
void ReLUActivationFunction(Eigen::Matrix<DataType, Dynamic, Dynamic> &aOutput)
{
    aOutput = aOutput.cwiseMax(0);
}
} // namespace dlfunctions

#endif