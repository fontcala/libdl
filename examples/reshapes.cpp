#include <iostream>
#include <libdl/dlfunctions.h>

using Eigen::MatrixXd;
int main()
{
    MatrixXd M(3, 3);
    M << 0, 1, 2,
        -1, 1, 2,
        -1, 1, 2;
    Eigen::Matrix<bool, Dynamic, Dynamic>  Mc = (M.array() > 0);
    std::cout << Mc << std::endl;
    MatrixXd some = M.array() * Mc.cast<double>().array();
    std::cout << some << std::endl;
    // const size_t vInputSampleNumber = 1;
    // const size_t vInputDepth1 = 3;
    // const size_t vInputHeight1 = 7;
    // const size_t vInputWidth1 = 5;
    // MatrixXd Input(vInputHeight1 * vInputWidth1, vInputDepth1);
    // Input << 0.680375, 0.0485744, 0.0632129, -0.211234, -0.012834, -0.921439, 0.566198, 0.94555, -0.124725, 0.59688, -0.414966, 0.86367, 0.823295, 0.542715, 0.86162, -0.604897, 0.05349, 0.441905, -0.329554, 0.539828, -0.431413, 0.536459, -0.199543, 0.477069, -0.444451, 0.783059, 0.279958, 0.10794, -0.433371, -0.291903 - 0.0452059, -0.295083, 0.375723, 0.257742, 0.615449, -0.668052, -0.270431, 0.838053, -0.119791, 0.0268018, -0.860489, 0.76015, 0.904459, 0.898654, 0.658402, 0.83239, 0.0519907, -0.339326, 0.271423, -0.827888, -0.542064, 0.434594, -0.615572, 0.786745, -0.716795, 0.326454, -0.29928, 0.213938, 0.780465, 0.37334, -0.967399, -0.302214, 0.912937, -0.514226, -0.871657, 0.17728, -0.725537, -0.959954, 0.314608, 0.608354, -0.0845965, 0.717353, -0.686642, -0.873808, -0.12088, -0.198111, -0.52344, 0.84794, -0.740419, 0.941268, -0.203127, -0.782382, 0.804416, 0.629534, 0.997849, 0.70184, 0.368437, -0.563486, -0.466669, 0.821944, 0.0258648, 0.0795207, -0.0350187, 0.678224, -0.249586, -0.56835, 0.22528, 0.520497, 0.900505, -0.407937, 0.0250707, 0.840257, 0.275105, 0.335448, -0.70468;
    // std::cout << "Input" << std::endl;
    // std::cout << Input << std::endl;

    // //Params
    // const size_t vFilterHeight1 = 5;
    // const size_t vFilterWidth1 = 2;
    // const size_t vPaddingHeight1 = 1;
    // const size_t vPaddingWidth1 = 1;
    // const size_t vStride1 = 2;

    // const size_t vOutputDepth1 = 6;
    // const size_t vOutputHeight1 = (vInputHeight1 - vFilterHeight1 + 2 * vPaddingHeight1) / vStride1 + 1;
    // const size_t vOutputWidth1 = (vInputWidth1 - vFilterWidth1 + 2 * vPaddingWidth1) / vStride1 + 1;

    // size_t vOutFields = vFilterHeight1 * vFilterWidth1 * vInputDepth1;
    // MatrixXd im2ColImage(vOutputHeight1 * vOutputWidth1, vOutFields);
    // dlfunctions::im2col(vFilterHeight1, vFilterWidth1, Input.data(), im2ColImage.data(), vOutputHeight1, vOutputWidth1, vOutFields, vInputHeight1, vInputWidth1, vInputDepth1,
    //                     vPaddingHeight1, vPaddingWidth1, vStride1, 1);
    // std::cout << "OutputVol 1" << std::endl;
    // std::cout << im2ColImage << std::endl;

    // MatrixXd col2ImImage = MatrixXd::Zero(vInputHeight1 * vInputWidth1, vInputDepth1);
    // dlfunctions::col2im(vFilterHeight1, vFilterWidth1, im2ColImage.data(), col2ImImage.data(), vOutputHeight1, vOutputWidth1, vOutFields, vInputHeight1, vInputWidth1, vInputDepth1,
    //         vPaddingHeight1, vPaddingWidth1, vStride1, 1);
    // std::cout << "OutputVol 3" << std::endl;
    // std::cout << col2ImImage << std::endl;

    // return 0;
}
