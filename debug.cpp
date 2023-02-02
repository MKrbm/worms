#include <autoobservable.hpp>
#include <iostream>

using namespace std;

int main(){
  std::cout << "hi" << std::endl;

  model::WormObservable wo(4, 2);
  cout << wo.get_state({1, 2, 3, 1}) << endl;

}