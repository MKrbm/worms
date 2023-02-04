#include <autoobservable.hpp>
#include <iostream>


#include <memory>
using namespace std;

int main(){
  std::cout << "hi" << std::endl;

  // model::ArrWormObs wo(2, 2);
  // cout << wo.GetState({1, 2, 3, 1}) << endl;
  // wo << 1.0;
  // batch_res res = wo.finalize();
  // cout << res.mean() << endl;
  // batch_obs& obs = wo;
  // wo << 3.0;
  // obs.operator<<(2.0);
  cout << alps::alea::is_alea_acc<batch_obs>::value << endl;
  cout << typeid(batch_obs::value_type).name() << endl;
  // cout << f.b.count() << endl;

  // wo << 2.0;
  // alps::alea::value_adapter<double> c = (double) 3;
  // wo << {c};
  

  // cout << obs.count() << endl;

}