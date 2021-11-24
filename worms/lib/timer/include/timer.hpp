#ifndef __time__
#define __time__

#include <iostream>

int sub_v2(int a, int b);

template<typename F>
auto wrapper_measure_time_v2(F func)
{
  std::cout << "this is inside the wrapper function v2" <<std::endl;
  auto return_func = [=]<typename... Args>(Args... args)
  {
    std::cout << "inside lambda" <<std::endl;

    return func(args...);
  };
  return return_func;
}

#endif