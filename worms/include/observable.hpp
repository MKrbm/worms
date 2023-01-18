#pragma once
#include <cmath> // for std::sqrt
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/access.hpp>
namespace BC {
  class observable {
  public:
    observable() : count_(0), sum_(0), esq_(0) {}
    void operator<<(double x) { sum_ += x; esq_ += x * x; ++count_; }
    int count() const {return count_; }
    double mean() const { return (count_ > 0) ? (sum_ / count_) : 0.; }
    double error(double r = 1) const {
      return (count_ > 1) ?
        std::sqrt((esq_ / count_ - mean() * mean()) / (count_*r - 1)) : 0.;
    }
    observable& operator+=(const observable& other) {
        sum_ += other.sum_;
        esq_ += other.esq_;
        count_ += other.count_;
        return *this;
    }
    observable operator+(const observable& other) const {
        observable result = *this;
        result += other;
        return result;
    }
  private:
    unsigned int count_;
    double sum_, esq_;

      // Boost serialization
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & count_;
        ar & sum_;
        ar & esq_;
    }
  };
} // end namespace bcl