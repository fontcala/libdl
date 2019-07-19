/** @file dlfunctions.h
 *  @author Adria Font Calvarons
 */

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
template <size_t FilterHeight, size_t FilterWidth>
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
/**
* Adapted from Caffe https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
* @warning To avoid nasty errors, make sure Matrix  \c *col is properly constructed(correct sizes). 
*/
template <class DataType>
void im2col(const size_t FilterHeight, const size_t FilterWidth, const DataType *img, DataType *col, size_t aOutHeight, size_t aOutWidth, size_t aOutFields, size_t height, size_t width,
            size_t pad_w, size_t pad_h, size_t aStride)
{
    //std::cout << FilterHeight << FilterWidth << aOutHeight << aOutWidth << aOutFields << height << width << channels<< "padw " << pad_w << "padh " << pad_h << "stride " << aStride << "samp " << aNumSamples << std::endl;
    for (size_t c = 0; c < aOutFields; ++c)
    {
        size_t w_offset = c % FilterWidth;
        size_t h_offset = (c / FilterWidth) % FilterHeight;
        size_t c_im = c / (FilterHeight * FilterWidth);
        for (size_t h = 0; h < aOutHeight; ++h)
        {
            size_t h_pad = h * aStride - pad_h + h_offset;
            for (size_t w = 0; w < aOutWidth; ++w)
            {
                size_t w_pad = w * aStride - pad_w + w_offset;
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

/**
* Adapted from Caffe https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
* @warning To avoid nasty errors, make sure Matrix  \c *col is properly constructed (correct sizes). 
* @warning To avoid wrong results, make sure Matrix  \c *col is initialized with zeros (this is not just a reshaping! Addition performed).
*/
template <class DataType>
void col2im(const size_t aFilterHeight, const size_t aFilterWidth, const DataType *aColData, DataType *aImData, size_t aOutHeight, size_t aOutWidth, size_t aOutFields,
            const size_t height, const size_t width,
            const size_t pad_w, const size_t pad_h,
            const size_t aStride)
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

template <class DataType>
Eigen::Matrix<DataType, Dynamic, Dynamic> flip(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aFilters, const size_t aNumberCuts)
{
    size_t vFilterSize = aFilters.rows();
    size_t vOutputDepth = aFilters.cols();
    size_t v2DFilterSize = vFilterSize / aNumberCuts;
    Eigen::Matrix<DataType, Dynamic, Dynamic> vToBeReturned = aFilters;

    for (size_t i = 0; i < aNumberCuts; ++i)
    {
        vToBeReturned.block(v2DFilterSize * i, 0, v2DFilterSize, vOutputDepth).colwise().reverseInPlace();
    }
    return vToBeReturned;
}

template <class DataType>
Eigen::Matrix<DataType, Dynamic, Dynamic> flatten(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput, const size_t aNumberCuts)
{
    size_t vInputDim1 = aInput.rows();
    size_t vInputDim2 = aInput.cols();
    size_t vBlockSize = vInputDim1 / aNumberCuts;
    size_t vOutputCols = vInputDim2 * vBlockSize;
    Eigen::Matrix<DataType, Dynamic, Dynamic> vToBeReturned(aNumberCuts, vOutputCols);

    for (size_t i = 0; i < aNumberCuts; ++i)
    {
        Eigen::Matrix<DataType, Dynamic, Dynamic> vSampleBlock = aInput.block(i * vBlockSize, 0, vBlockSize, vInputDim2);
        vSampleBlock.resize(1, vOutputCols);
        vToBeReturned.row(i) = vSampleBlock;
    }
    return vToBeReturned;
}

template <class DataType>
Eigen::Matrix<DataType, Dynamic, Dynamic> unflatten(const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInput, const size_t aInputDepth, const size_t aInputHeight, const size_t aInputWidth)
{
    size_t vNumberSamples = aInput.rows();
    size_t vUnflatSampleSize = aInputHeight * aInputWidth / vNumberSamples;
    Eigen::Matrix<DataType, Dynamic, Dynamic> vToBeReturned(vNumberSamples * vUnflatSampleSize, aInputDepth);
    for (size_t i = 0; i < vNumberSamples; ++i)
    {
        Eigen::Matrix<DataType, Dynamic, Dynamic> vSampleBlock = aInput.row(i);
        vSampleBlock.resize(vUnflatSampleSize, aInputDepth);
        vToBeReturned.block(i * vUnflatSampleSize, 0, vUnflatSampleSize, aInputDepth) = vSampleBlock;
    }
    return vToBeReturned;
}

template <class DataType>
void convolution(Eigen::Matrix<DataType, Dynamic, Dynamic> &aConvolutedOutput, size_t aOutHeight, size_t aOutWidth, const Eigen::Matrix<DataType, Dynamic, Dynamic> &aFilters, const size_t aFilterHeight, const size_t aFilterWidth, const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInputImage, size_t height, size_t width, size_t channels,
                 size_t pad_w, size_t pad_h, size_t aStride, size_t aNumSamples)
{
    size_t vOutFields = aFilterHeight * aFilterWidth * channels;
    Eigen::Matrix<DataType, Dynamic, Dynamic> im2ColImage(aOutHeight * aOutWidth, vOutFields);
    dlfunctions::im2col(aFilterHeight, aFilterWidth, aInputImage.data(), im2ColImage.data(), aOutHeight, aOutWidth, vOutFields, height, width, pad_w, pad_h, aStride);
    aConvolutedOutput = im2ColImage * aFilters;
}

template <class DataType>
void fullconvolution(Eigen::Matrix<DataType, Dynamic, Dynamic> &aConvolutedOutput, size_t aOutHeight, size_t aOutWidth, const Eigen::Matrix<DataType, Dynamic, Dynamic> &aFilters, const size_t aFilterHeight, const size_t aFilterWidth, const Eigen::Matrix<DataType, Dynamic, Dynamic> &aInputImage, size_t height, size_t width, size_t channels,
                     size_t aStride, size_t aNumSamples)
{
    size_t vPadHeight = aFilterHeight - 1;
    size_t vPadWidth = aFilterWidth - 1;
    dlfunctions::convolution(aConvolutedOutput, aOutHeight, aOutWidth, aFilters, aFilterHeight, aFilterWidth, aInputImage, height, width, vPadHeight, vPadWidth, aStride, aNumSamples);
}

// Based on im2col, only 2D. TODO: Can be made more efficient
template <class DataType>
void im2colpool(const size_t PoolSize, const DataType *img, DataType *col, size_t aOutHeight, size_t aOutWidth, size_t height, size_t width, size_t aStride, size_t aNumSamples)
{
    const size_t vOutFields = PoolSize * PoolSize;
    for (size_t c = 0; c < vOutFields; ++c)
    {
        size_t w_offset = c % PoolSize;
        size_t h_offset = (c / PoolSize) % PoolSize;
        size_t c_im = c / (PoolSize * PoolSize);
        for (size_t h = 0; h < aOutHeight; ++h)
        {
            size_t h_pad = h * aStride + h_offset;
            for (size_t w = 0; w < aOutWidth; ++w)
            {
                size_t w_pad = w * aStride + w_offset;
                col[(c * aOutHeight + h) * aOutWidth + w] =
                    img[(c_im * height + h_pad) * width + w_pad];
            }
        }
    }
}

template <class DataType>
Eigen::Matrix<DataType, Dynamic, 1> im2pool(const size_t PoolSize, const DataType *img, size_t aOutHeight, size_t aOutWidth, size_t height, size_t width, size_t aStride, size_t aNumSamples)
{
    const size_t vOutFields = PoolSize * PoolSize;
    Eigen::Matrix<DataType, Dynamic, Dynamic> PrePool(aOutHeight * aOutWidth, vOutFields);
    im2colpool(PoolSize, img, PrePool.data(), aOutHeight, aOutWidth, height, width, aStride, aNumSamples);
    Eigen::Matrix<DataType, Dynamic, 1> PostPool = PrePool.rowwise().maxCoeff();
    return PostPool;
}

template <class DataType>
void colpool2im(const size_t aPoolSize, const DataType *aColData, DataType *aImData, size_t aOutHeight, size_t aOutWidth, size_t aOutFields,
                const size_t height, const size_t width,
                const size_t aStride, const size_t aNumSamples)
{
    for (size_t c = 0; c < aOutFields; ++c)
    {
        size_t w_offset = c % aPoolSize;
        size_t h_offset = (c / aPoolSize) % aPoolSize;
        size_t c_im = c / aPoolSize / aPoolSize;
        for (size_t h = 0; h < aOutHeight; ++h)
        {
            size_t h_pad = h * aStride + h_offset;
            for (size_t w = 0; w < aOutWidth; ++w)
            {
                size_t w_pad = w * aStride + w_offset;
                aImData[(c_im * height + h_pad) * width + w_pad] +=
                    aColData[(c * aOutHeight + h) * aOutWidth + w];
            }
        }
    }
};

} // namespace dlfunctions

#endif