#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include "catch2/catch.hpp"
#include <string>
#include <iostream>
#include <Eigen/Core>
#include "libdl/dlfunctions.h"
#include <spdlog/spdlog.h>

// catch2
TEST_CASE("sampled AMM all samples equals normal multiplication", "experimental")
{
    
    // eigen
    using Eigen::MatrixXd;

    std::random_device rd;
    std::mt19937 vRandom(rd());
    std::normal_distribution<float> vRandDistr(0, sqrt(static_cast<double>(2) / static_cast<double>(12)));
    
    MatrixXd mR = MatrixXd::NullaryExpr(5,4, [&]() { return vRandDistr(vRandom); });
    std::cout << "mR" << std::endl;
    std::cout << mR << std::endl;
    
    MatrixXd mB = MatrixXd::NullaryExpr(4,5, [&]() { return vRandDistr(vRandom); });
    std::cout << "mB" << std::endl;
    std::cout << mB << std::endl;
    //
    MatrixXd sampled = dlfunctions::topkAMM(mR,mB,1);
    std::cout << "sampled" << std::endl;
    std::cout << sampled << std::endl;
    MatrixXd real = mR * mB;
    std::cout << "real" << std::endl;
    std::cout << real << std::endl;


    MatrixXd mR2 = MatrixXd::NullaryExpr(36,14, [&]() { return vRandDistr(vRandom); });
    MatrixXd mB2 = MatrixXd::NullaryExpr(14,51, [&]() { return vRandDistr(vRandom); });

    MatrixXd sampled2 = dlfunctions::topkAMM(mR2,mB2,1);
    MatrixXd real2 = mR2 * mB2;

    REQUIRE(sampled.sum() == Approx(real.sum()));
    REQUIRE(sampled2.sum() == Approx(real2.sum()));
}
