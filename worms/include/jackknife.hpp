#pragma once
#include <alps/alea.hpp>
#include <alps/alea/util/model.hpp>

#include <Eigen/Dense>

#pragma once

#include <alps/alea/core.hpp>



namespace alps { namespace alea {
template <typename T> struct scalar_trinary_transformer;


template <typename T>
scalar_trinary_transformer<T> make_transformer(std::function<T(T,T,T)> fn)
{
    return scalar_trinary_transformer<T>(fn);
}

template <typename T>
struct scalar_trinary_transformer
    : public transformer<T>
{
public:
    scalar_trinary_transformer(const std::function<T(T,T,T)> &fn) : fn_(fn) { }

    size_t in_size() const { return 3; }

    size_t out_size() const { return 3; }

    column<T> operator() (const column<T> &in) const
    {
        if (in.size() != in_size())
            throw size_mismatch();

        column<T> ret(1);
        ret(0) = fn_(in(0), in(1), in(2));
        return ret;
    }

private:
    std::function<T(T,T,T)> fn_;
};
// end of class declaration



//* define jackknife transform


// estimator is simple mean
/*
params
------
obs : batch_result<double>  
    batch_result object containing an observable. May be an average sign

return 
------
mean and error estimated by jackknife

*/
std::pair<double, double> jackknife_reweight_single(alea::batch_result<double> obs){
  
  // define transformer
  auto f = [] (double x) -> double { return x; };
  alea::batch_result<double> prop = alea::transform(
                  alea::jackknife_prop(),
                  alea::make_transformer(std::function<double(double)>(f)),
                  obs
                  );
  return std::make_pair((double)prop.mean()[0], (double)prop.stderror()[0]);
}


// estimator is division of two mean.
/*
params
------
obs : batch_result<double>  
    batch_result object containing the observed data.

as : batch_result<double>  
    batch_result object containing the average sign.

return 
------
mean and error for a model with negative sign problem estimated by jackknife

*/
std::pair<double, double> jackknife_reweight_div(alea::batch_result<double> obs, alea::batch_result<double> as){
  
  // define join batch (you can also define join_rbatch by appending another dimension to batch_result<T>)
  alea::batch_result<double> join_rbatch = alps::alea::join(obs, as); 

  // define transformer
  // esitimator is f(x,y) = x/y
  auto f = [] (double x, double y) -> double { return x/y; };
  alea::batch_result<double> prop = alea::transform(
                  alea::jackknife_prop(),
                  alea::make_transformer(std::function<double(double, double)>(f)),
                  join_rbatch
                  );
  return std::make_pair((double)prop.mean()[0], (double)prop.stderror()[0]);
}



/*

receive function as a parameter. Use two observables and one average sign to esitimate the mean and error of a statistic.
params
------
obs1 : batch_result<double>  
    batch_result object containing the first observed data.

obs2 : batch_result<double>
    batch_result object containing the second observed data.

as : batch_result<double>  
    batch_result object containing the average sign.

return 
------
mean and error for a model with negative sign problem estimated by jackknife

*/
std::pair<double, double> jackknife_reweight_any(
                    alea::batch_result<double> obs1, 
                    alea::batch_result<double> obs2, 
                    alea::batch_result<double> as,
                    std::function<double(double, double, double)> f
){
  
  // define join batch (you can also define join_rbatch by appending another dimension to batch_result<T>)
  alea::batch_result<double> join_rbatch = alps::alea::join(obs1, obs2); 
  join_rbatch = alps::alea::join(join_rbatch, as);
  alea::batch_result<double> prop = alea::transform(
                  alea::jackknife_prop(),
                  alea::make_transformer(std::function<double(double, double, double)>(f)),
                  join_rbatch
                  );
  return std::make_pair((double)prop.mean()[0], (double)prop.stderror()[0]);
}

}}


