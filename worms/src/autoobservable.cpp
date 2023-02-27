#include <fstream>

#include "../include/autoobservable.hpp"
#include "load_npy.hpp"
#include "../include/funcs.hpp"
#include <alps/alea/computed.hpp>
using namespace std;

bool endsWith(const string &str, const string &suffix)
{
  return str.size() >= suffix.size() &&
         str.rfind(suffix) == str.size() - suffix.size();
}

bool fileExists(const string &path)
{
  ifstream file(path);
  return file.good();
}

namespace model
{
  BaseWormObs::BaseWormObs(size_t spin_dof, size_t legs) : _spin_dof(spin_dof), _leg_size(legs)
  {
    if (legs != 1 && legs != 2)
    {
      std::cerr << "leg size of BaseWormObs must be 1 or 2" << std::endl;
      exit(1);
    }
  }

  int BaseWormObs::GetState(size_t x1,
                            size_t x2,
                            size_t x3,
                            size_t x4)
      const
  {
    if (x1 >= _spin_dof || x2 >= _spin_dof || x2 >= _spin_dof || x3 >= _spin_dof)
    {
      std::cerr << "spin state is out of range" << std::endl;
      exit(1);
    }
    if (is_one_site())
    {
      std::cerr << "This is one site operator, but trying to call bond operator." << std::endl;
      exit(1);
    }
    size_t s = 0;
    s += x4;
    s *= _spin_dof;
    s += x3;
    s *= _spin_dof;
    s += x2;
    s *= _spin_dof;
    s += x1;
    // size_t s = 0;
    // for (auto& spin : spins) {
    //   s *= _spin_dof;
    //   s += spin;
    // }
    // std::array<size_t, 4>::iterator si = spins.end();
    // do {
    //   si--;
    //   s *= _spin_dof;
    //   s += *si;
    // } while (si != spins.begin());
    return s;
  }

  int BaseWormObs::GetState(size_t x1, size_t x2) const
  {
    if (x1 >= _spin_dof || x2 >= _spin_dof)
    {
      std::cerr << "spin state is out of range" << std::endl;
      exit(1);
    }
    if (!is_one_site())
    {
      std::cerr << "This is bond operator, but trying to call one site operator." << std::endl;
      exit(1);
    }
    return x1 + x2 * _spin_dof;
  }

  double BaseWormObs::operator()(std::array<size_t, 4> spins, double r, double tau) const
  {
    cerr << "This method is not implemented" << endl;
    if (!has_operator())
    {
      std::runtime_error("operator is not set yet");
    }
  }

  double BaseWormObs::operator()(size_t x1,
                                 size_t x2,
                                 size_t x3,
                                 size_t x4) const
  {
    if (!has_operator())
    {
      std::runtime_error("operator is not set yet");
    }
    if (is_one_site())
    {
      std::runtime_error("This is one site operator, but tring to call bond operator.");
    }
    int s = GetState(x1, x2, x3, x4);
    return _operator(s);
  }

  double BaseWormObs::operator()(size_t x1, size_t x2) const
  {
    if (!has_operator())
    {
      std::runtime_error("operator is not set yet");
    }
    if (!is_one_site())
    {
      std::runtime_error("This is bond operator, but tring to call one site operator.");
    }
    int s = GetState(x1, x2);
    return _operator(s);
  }

  bool BaseWormObs::check_is_symm() const
  {
    if (!has_operator())
    {
      std::runtime_error("operator is not set yet");
    }
    if (is_one_site())
    {
      return true;
    }
    else
    { // 2 points operator
      for (size_t _idx = 0; _idx < round(pow(_spin_dof, 2 * 2)); _idx++)
      {
        size_t idx = _idx;
        size_t t_x = idx % _spin_dof;
        idx /= _spin_dof;
        size_t t_x_prime = idx % _spin_dof;
        idx /= _spin_dof;
        size_t h_x = idx % _spin_dof;
        idx /= _spin_dof;
        size_t h_x_prime = idx % _spin_dof;
        if (operator()(h_x, t_x, h_x_prime, t_x_prime) != operator()(t_x, h_x, t_x_prime, h_x_prime))
        {
          return false;
        }
      }
      return true;
    }
  }

  bool BaseWormObs::check_no_single() const
  {
    if (!has_operator())
    {
      std::runtime_error("operator is not set yet");
    }
    if (is_one_site())
    {
      return true;
    }
    else
    {
      for (int _idx = 0; _idx < round(pow(_spin_dof, 2 * 3)); _idx++)
      {
        size_t idx = _idx;
        size_t i = idx % _spin_dof;
        idx /= _spin_dof;
        size_t j = idx % _spin_dof;
        idx /= _spin_dof;
        size_t k = idx % _spin_dof;
        idx /= _spin_dof;
        if (operator()(i, j, i, k) != 0 || operator()(j, i, k, i) != 0)
          return false;
      }
    }
    return true;
  }

  bool BaseWormObs::check_no_diagonal() const
  {
    if (!has_operator())
    {
      std::runtime_error("operator is not set yet");
    }
    if (is_one_site())
    {
      return true;
    }
    else
    {
      for (size_t i = 0; i < _spin_dof; i++)
      {
        for (size_t j = 0; j < _spin_dof; j++)
        {
          if (operator()(i, j, i, j) != 0)
            return false;
        }
      }
    }
    return true;
  }

  double BaseWormObs::_operator(size_t state) const
  {
    std::cerr << "BaseWormObs::operator is virtual function" << std::endl;
    exit(1);
  }

  ArrWormObs::ArrWormObs(size_t sdof, size_t legs) : BaseWormObs(sdof, legs) {}

  void ArrWormObs::_SetWormObs(vector<double> _obs, bool print)
  {
    if (_worm_obs.size() != 0 )
    {
      if (print) cerr << "Worm_obs is already set" << endl;
      return;
    }
    if (pow(_spin_dof, _leg_size * 2) != _obs.size())
    {
      throw std::runtime_error("dimension dissmatch");
    }
    _worm_obs = move(_obs);
    _has_operator = true;
    if (!check_is_symm())
    {
      if (print) cerr << "Warning!! Given array is not symmetric under the swap" << endl;
    }

    if (!check_no_single())
    {
      if (print) cerr << "Warning!! Given array has non-zero single site operator (Cannot handle yet)" << endl;
      if (print) cout << "If your hamiltonian doesn't have single site operator, this cause problem." << endl;
    }

    // if(!check_no_diagonal()){
    //   cout << "Warning!! Given array has non-zero diagonal operator (Cannot handle yet)" << endl;
    // }
  }

  // read obs vector from numpy file.
  /*
              x_t'         x_h'
  weight of   2 points ops (O)     =   <x' | O | x >   = O_{x', x}
              x_t          x_h

  x = x_t + x_h * S
  {x', x} -> x' * S^2  + x

  */
  void ArrWormObs::ReadNpy(string file_path, bool print)
  {
    if (!endsWith(file_path, ".npy"))
    {
      cerr << "File path is not *.npy" << endl;
      exit(1);
    }

    if (!fileExists(file_path))
    {
      cerr << "File does not exist" << endl;
      exit(1);
    }

    pair<vector<size_t>, vector<double>> pair;
    try
    {
      pair = load_npy(file_path);
    }
    catch (const std::exception &e)
    {
      cerr << "Fail to load npy file with unknown reason" << endl;
      exit(1);
    }
    vector<size_t> shape = pair.first;
    vector<double> data = pair.second;

    if (shape.size() != 2)
    {
      cerr << "Require 2D array" << endl;
      exit(1);
    }

    _SetWormObs(data, print);
    // cerr << "Array loaded from : " << file_path << endl;

    return;
  }

  double ArrWormObs::_operator(size_t state) const
  {
    return _worm_obs[state];
  }

  WormObs::WormObs(size_t spin_dof, std::string folder_path, bool print) : WormObs(spin_dof, move(ReadNpy(spin_dof, folder_path, print)))
  {
    // cout <<(*obs1)({1,0,0,1}) << endl;
    // cerr << "WormObs is not implemented yet" << endl;
    // _worm_obs_ptr.first = make_unique<BaseWormObs>(spin_dof, 1);
    // _worm_obs_ptr.second = make_unique<BaseWormObs>(spin_dof, 2);
  }

  WormObs::WormObs(size_t spin_dof, pair<BaseWormObsPtr, BaseWormObsPtr> obspt_pair) : _spin_dof(spin_dof), batch_obs(1)
  {
    BaseWormObsPtr obs1 = move(obspt_pair.first);
    BaseWormObsPtr obs2 = move(obspt_pair.second);
    if (obs1->leg_size() != 1 || obs2->leg_size() != 2)
    {
      cerr << "WormObs requires 1 and 2 leg operators" << endl;
      exit(1);
    }

    if (obs1->spin_dof() != spin_dof || obs2->spin_dof() != spin_dof)
    {
      cerr << "Spin dof of operators does not match" << endl;
      exit(1);
    }

    if (!obs1->has_operator() || !obs2->has_operator())
    {
      cerr << "Operators are not set yet" << endl;
      exit(1);
    }
    _first = move(obs1);
    _second = move(obs2);
  }

  /*
 spins = {x_t, x_h, x_t_p, x_h_p}
  */
  void WormObs::add(std::array<size_t, 4> spins, size_t L, int sign, double r, double tau)
  {
    BaseWormObs &obs_1site = *(_first);  // one site worm observable
    BaseWormObs &obs_2site = *(_second); // two site worm observable

    // if (obs_2site(spins))
    *this << (obs_1site(spins[0], spins[1]) + obs_2site(spins[0], spins[1], spins[2], spins[3]) / 2) * L * sign;

    // dout << "spins : " << spins[0] << " " << spins[1] << " " << spins[2] << " " << spins[3] <<"\t";
    // dout << "add : " << obs_1site(spins)* L * sign << " " << obs_2site(spins)* L * sign/2 << endl;
    // dout << " obs sum : " << this->store().batch().rowwise().sum() << endl;
    // cout << obs_2site(spins) << endl;
    // obs_1site << (double)(sign * obs_1site(spins) * L);
    // obs_2site << (double)(sign * obs_2site(spins)* L / 2); // independent of r and tau currently.
  }

  typedef shared_ptr<BaseWormObs> BaseWormObsPtr;
  pair<BaseWormObsPtr, BaseWormObsPtr> WormObs::ReadNpy(size_t spin_dof, std::string folder_path, bool print)
  {
    ArrWormObs awo1 = ArrWormObs(spin_dof, 1);
    ArrWormObs awo2 = ArrWormObs(spin_dof, 2);

    if (folder_path == "")
    {
      // set zero operator to all.
      awo1._SetWormObs(vector<double>(powl(spin_dof, 2)), print);
      awo2._SetWormObs(vector<double>(powl(spin_dof, 4)), print);
      return make_pair(make_shared<ArrWormObs>(awo1), make_shared<ArrWormObs>(awo2));
    }

    std::vector<std::string> paths;
    try
    {
      get_npy_path(folder_path, paths);
    }
    catch (const std::exception &e)
    {
      std::cerr << e.what() << '\n';
      exit(1);
    }

    if (paths.size() == 0 | paths.size() > 2)
    {
      cerr << "Require 1 or 2 numpy files in the folder" << endl;
      exit(1);
    }
    std::sort(paths.begin(), paths.end());

    if (paths.size() == 2)
    {
      try
      {
        awo1.ReadNpy(paths[0], print);
        awo2.ReadNpy(paths[1], print);
        if (print)
          cout << "wobs is read from files : " << paths[0] << " and " << paths[1] << endl;
      }
      catch (const std::exception &e)
      {
        // cerr << "Warning! : " << e.what() << '\n';
        try
        {
          awo1.ReadNpy(paths[1], print);
          awo2.ReadNpy(paths[0], print);
          if(print) cerr << "Warning! : Paths might be in reverse order " << endl;
          if (print)
            cout << "wobs is read from files : " << paths[1] << " and " << paths[0] << endl;
        }
        catch (const std::exception &e)
        {
          cerr << e.what() << '\n';
          cerr << "Fail to read numpy files." << endl;
          exit(1);
        }
      }
    }
    else
    {
      try
      {
        awo1.ReadNpy(paths[0], print);
        awo2._SetWormObs(vector<double>(powl(spin_dof, 4)), print);
        if (print){
          cerr << "Warning! : Only one numpy file is found. The path is set for 1 points operator " << endl;
          cout << "wobs is read from files : " << paths[0] << endl;
        }
      }
      catch (const std::exception &e)
      {
        try
        {
          awo1._SetWormObs(vector<double>(powl(spin_dof, 2)), print);
          awo2.ReadNpy(paths[0], print);
          if (print)
          {
            cerr << "Warning! : Only one numpy file is found. The path is set for 2 points operator " << endl;
            cout << "wobs is read from files : " << paths[0] << endl;
          }
        }
        catch (const std::exception &e)
        {
          cout << e.what() << '\n';
          exit(1);
        }
      }
    }

    // somehow this works fine but if I used make_shared<BaseWormObs>(awo1) it does not work.
    return make_pair(make_shared<ArrWormObs>(awo1), make_shared<ArrWormObs>(awo2));
    // return make_pair(sp1, sp2);
  }

  template<typename... Args>
  MapWormObs::MapWormObs(std::pair<std::string, WormObs> arg, Args... args )
  :MapWormObs(args...)
  {
    _worm_obs_map[arg.first] = arg.second;
  }
  MapWormObs::MapWormObs(std::pair<std::string, WormObs> arg)
  {
    _worm_obs_map[arg.first] = arg.second;
  }

  template MapWormObs::MapWormObs(worm_pair_t, worm_pair_t);
  template MapWormObs::MapWormObs(worm_pair_t, worm_pair_t, worm_pair_t);
  template MapWormObs::MapWormObs(worm_pair_t, worm_pair_t,worm_pair_t, worm_pair_t);
}
