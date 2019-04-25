#ifndef HELLO_H
#define HELLO_H

#include <string>
#include <iostream>
void hello(std::string const& name)
{
  std::cout << "Hello, " << name << "!\n";
}

#endif
