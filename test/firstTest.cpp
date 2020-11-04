#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include "catch2/catch.hpp"
#include <string>
#include <iostream>
#include <Eigen/Core>
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
    std::cout << mR << std::endl;
    mR = MatrixXd::NullaryExpr(5,4, [&]() { return vRandDistr(vRandom); });
    std::cout << mR << std::endl;
    //C++17
    std::string name = "Hello world";
    if (const auto it = name.find("Hello"); it != std::string::npos)
        std::cout << it << " Hello\n";
    std::cout << "Hello, " << name << "!\n";
}
