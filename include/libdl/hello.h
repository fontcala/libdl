#ifndef HELLO_H
#define HELLO_H

#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <spdlog/spdlog.h>
#include <pybind11/pybind11.h>

using Eigen::MatrixXd;

int hello(std::string const& name)
{
  std::cout << "test includes: " << std::endl;
  spdlog::info("Hello, {}!", "World");
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;
  std::cout << "Hello, " << name << "!\n";

  return 0;
}

#endif