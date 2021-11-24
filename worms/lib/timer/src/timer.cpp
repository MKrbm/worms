#include "../include/timer.hpp"

int sub(int a, int b){
  return a-b;
}

auto sub_v1 = wrapper_measure_time_v2(sub);

int sub_v2(int a, int b){
  return sub_v1(a,b);
}