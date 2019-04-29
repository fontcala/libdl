/** @file hello.h
 * @author Adria Font Calvarons
 */

#ifndef HELLO_H
#define HELLO_H

#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <spdlog/spdlog.h>

using Eigen::MatrixXd;

/** hello function
    @param name
    @return 0
*/
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
  //cpp17
  if (const auto it = name.find("Hello"); it != std::string::npos)
    std::cout << it << " Hello\n";
  std::cout << "Hello, " << name << "!\n";

  return 0;
}

#endif
