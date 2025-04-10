#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include "catch2/catch.hpp"
#include <string>
#include <iostream>
#include <Eigen/Core>
#include "libdl/dlfunctions.h"
#include <spdlog/spdlog.h>

// catch2
TEST_CASE("all library submoduels working and c++17 are available", "[includes]")
{
    
    // spdlog
    spdlog::info("Hello, {}!", "World");

    // eigen
    using Eigen::MatrixXd;
    MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);
    m = m.array().sign();
    std::cout << m << std::endl;

    //
    std::random_device rd;
    std::mt19937 vRandom(rd());
    std::normal_distribution<float> vRandDistr(0, sqrt(static_cast<double>(2) / static_cast<double>(12)));
    MatrixXd mR = MatrixXd::NullaryExpr(5,4, [&]() { return vRandDistr(vRandom); });
    std::cout << "mR" << std::endl;
    std::cout << mR << std::endl;
    mR = MatrixXd::NullaryExpr(5,4, [&]() { return vRandDistr(vRandom); });
    MatrixXd mB = MatrixXd::NullaryExpr(4,5, [&]() { return vRandDistr(vRandom); });
    std::cout << "mB" << std::endl;
    std::cout << mB << std::endl;
    std::vector<int> ind{1,3};
    MatrixXd subMatrix = mR(ind,Eigen::all);
    std::cout << "submatrix" << std::endl;
    std::cout << subMatrix << std::endl;
    //
    MatrixXd sampled = dlfunctions::topkAMM(mR,mB,1);
    std::cout << "sampled" << std::endl;
    std::cout << sampled << std::endl;
    std::cout << "real" << std::endl;
    std::cout << mR * mB << std::endl;


    // Aproximate multiplication
    //C++17
    std::string name = "Hello world";
    if (const auto it = name.find("Hello"); it != std::string::npos)
        std::cout << it << " Hello\n";
    std::cout << "Hello, " << name << "!\n";
}
