#ifndef DLFUNCTIONS_H
#define DLFUNCTIONS_H

#include <random>
#include <stdexcept>
#include <string>
#include <iostream>
#include <Eigen/Core>
//#include <spdlog/spdlog.h>

using Eigen::MatrixXd;

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
template <int FilterHeight, int FilterWidth>
void im2col(const double *img, double *col, size_t aOutHeight, size_t aOutWidth, size_t aOutFields, int width, int height, int channels,
            int pad_w, int pad_h, int aStride, size_t aNumSamples)
{
    int imOffset = aOutHeight * aOutWidth * aOutFields;
    for (size_t vSample = 0; vSample < aNumSamples; vSample++)
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
                        col[vSample * (imOffset - 1) + (c * aOutHeight + h) * aOutWidth + w] =
                            img[vSample * (imOffset - 1) + (c_im * height + h_pad) * width + w_pad];
                    }
                    else
                    {
                        col[vSample * (imOffset - 1) + (c * aOutHeight + h) * aOutWidth + w] = 0;
                    }
                }
            }
        }
    }
}

template <int FilterHeight, int FilterWidth>
void convolution(MatrixXd& aConvolutedOutput, const MatrixXd &aFilters, const MatrixXd& aInputImage, size_t aOutHeight, size_t aOutWidth, int width, int height, int channels,
                 int pad_w, int pad_h, int aStride, size_t aNumSamples)
{
    size_t vOutFields = FilterHeight * FilterWidth * channels;
    MatrixXd im2ColImage(aOutHeight * aOutWidth, vOutFields);

    im2col<FilterHeight,FilterWidth>(aInputImage.data(), im2ColImage.data(), aOutHeight, aOutWidth, vOutFields, width, height, channels, pad_w, pad_h, aStride, aNumSamples);

    aConvolutedOutput = im2ColImage * aFilters;
}
void ReLUActivationFunction(MatrixXd &aOutput)
{
    aOutput = aOutput.cwiseMax(0);
}
} // namespace dlfunctions

#endif