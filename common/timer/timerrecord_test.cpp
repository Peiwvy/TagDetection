#include "timer.h"

#include <iostream>

void func1() {
  for (int i = 0; i < 100000000; i++)
    ;
}

void func2() {
  for (int i = 0; i < 200000000; i++)
    ;
}

int main() {
  Timer::Evaluate(func1, "Function 1");
  Timer::Evaluate(func1, "Function 1");
  Timer::Evaluate(func1, "Function 1");

  Timer::Evaluate(func2, "Function 2");
  Timer::Evaluate(func2, "Function 2");
  Timer::Evaluate(func2, "Function 2");

  Timer::Evaluate(
    []() {
      for (int i = 0; i < 300000000; i++)
        ;
    },
    "Function 3");

  Timer::PrintAll();

  std::cout << "Mean time for Function 1: " << Timer::GetMeanTime("Function 1") << "ms" << std::endl;
  std::cout << "Mean time for Function 2: " << Timer::GetMeanTime("Function 2") << "ms" << std::endl;
  std::cout << "Mean time for Function 3: " << Timer::GetMeanTime("Function 3") << "ms" << std::endl;

  Timer::DumpIntoFile("timer.log");

  Timer::Clear();

  return 0;
}
