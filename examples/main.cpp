/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <random>
#include <iostream>

#include <alps/alea.hpp>
#include <alps/alea/util/model.hpp>

#include <Eigen/Dense>

#include <typeinfo>
#include <typeindex>

using namespace alps;
using namespace std;

template<class T> std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[ ";
    for ( const T& item : vec )
        os << item << ", ";
    os << "]"; return os;
}

alea::util::var1_model<double> create_model(double x = 0.97)
{
    Eigen::VectorXd phi0(1), eps(1);
    Eigen::MatrixXd phi1(1,1);

    phi0 << 3.0 * x;
    phi1 << x;
    eps << 2.0;
    return alea::util::var1_model<double>(phi0, phi1, eps);
}
int main()
{
    // Construct a data set from a very simple prescription called VAR(1).
    // A detailed explanation of VAR(P) models can be found in
    // https://www.kevinsheppard.com/images/5/56/Chapter5.pdf
    alea::util::var1_model<double> model = create_model(0.3);
    alea::util::var1_model<double> model2 = create_model(0.97);
    std::cout << "Exact <X> =" << model.mean().transpose() << "\n";
    std::cout << "Exact autocorr. time = " << model.ctau() << "\n";

    // Set up two accumulators: one which tries to estimate autocorrelation
    // time and one which keeps track of the distribution of values.  The "1"
    // denotes the number of vector components (1).
    alea::autocorr_acc<double> acorr(1);
    alea::batch_acc<double> abatch(1, 500);
    alea::batch_acc<double> abatch2(1, 500);


    // Set up random number generator
    std::mt19937 prng1(0);
    std::mt19937 prng2(2);


    alea::value_adapter<double> x = 1000.0;

    // Generate data points and add them to the accumulators
    std::cout << "\nSimulation data:\n";
    alea::util::var1_run<double> generator1 = model.start();
    alea::util::var1_run<double> generator2 = model2.start();

    while(generator1.t() < 100+5000) {
        // Measure the current value
        double current = generator1.xt()[0];
        double current2 = generator2.xt()[0];
        generator1.step(prng1);
        generator2.step(prng2);
        // std::vector<double> v = {current, current2};
        if (generator1.t() < 5000 ) continue;

        // Add current data point to the accumulator.  Both accumulators
        // have very little runtime overhead.
        acorr << current;
        abatch << current;
        abatch2 << current2;

        // Perform step
        // generator1.step(prng1);
        // generator2.step(prng2);
    }

    // Analyze data: we repurpose our accumulated data as results by calling
    // finalize().  Before we can accumulate again, we would need to call reset()
    // on the accumulators
    alea::autocorr_result<double> rcorr = acorr.finalize();
    alea::batch_result<double> rbatch = abatch.finalize();
    alea::batch_result<double> rbatch2 = abatch2.finalize();
    alea::batch_result<double> join_rbatch = alps::alea::join(rbatch, rbatch2);

    cout << join_rbatch.mean() << endl;
    // alea::batch_result<vector<double>> vec_mean = joined_rbatch;

    // cout << vec_mean << endl;


    // The autocorrelation accumulator measures the autocorrelation time and
    // corrects the standard error of the mean accordingly.
    std::cout << "Measured <X> = " << rcorr << "\n";
    std::cout << "Measured autocorr. time = " << rcorr.tau() << "\n";

    // Compare result to analytic result using hypothesis testing
    alea::t2_result test = alea::test_mean(rcorr, model.mean());
    std::cout << "p-value = " << test.pvalue() << "\n";
    if (test.pvalue() >= 0.05)
        std::cout << "Results are consistent at the alpha = 0.05 level\n";

    // std::cout << "Type: " << typeid(joined_res).name() << std::endl;
    // std::cout << "Type: " << std::type_index(typeid(joined_res)).name() << std::endl;

    // Estimate <1/x> by performing a Jackknife error propagation
    std::cout << "\nError propagation:\n";
    auto f = [] (double x, double y) -> double { return y/x; };
    alea::batch_result<double> prop = alea::transform(
                    alea::jackknife_prop(),
                    alea::make_transformer(std::function<double(double, double)>(f)),
                    join_rbatch
                    );

    auto g = [] (double x) -> double { return x; };
    alea::batch_result<double> prop1 = alea::transform(
                    alea::jackknife_prop(),
                    alea::make_transformer(std::function<double(double)>(g)),
                    rbatch
                    );
    alea::batch_result<double> prop2 = alea::transform(
                    alea::jackknife_prop(),
                    alea::make_transformer(std::function<double(double)>(g)),
                    rbatch2
                    );
    std::cout << "Measured <Y/X> = " << prop << "\n";
    std::cout << "Measured <Y/X> = " << ((double) prop2.mean()[0]) / ((double) prop1.mean()[0]) << "\n";
    std::cout << "Exact <Y/X> = " << model2.mean()[0] / model.mean()[0] << endl;

    // alea::t2_result test2 = alea::test_mean(prop, model.mean().cwiseInverse());
    // std::cout << "p-value = " << test2.pvalue() << "\n";
    return 0;
}
