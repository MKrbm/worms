#include <iostream>
#include <fstream>
#include <stdio.h>
#include <chrono>
#include <memory>
#include <iterator>
#include <cstdio>
// #include <gmpxx.h>

using namespace std;
using namespace std::chrono;
using std::cout;
using std::endl;
using std::ofstream;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
using std::chrono::microseconds;


int add(int a, int b){
  return a+b;
}

inline int modifyBit(int n, int p, int b)
{
    return ((n & ~(1 << p)) | (b << p));
}

inline int getbit(int n, int p)
{
    return (n >> p) ^ 1;
}

template <unsigned int L = 2>
class binstate{
public:
  int x;
  binstate(){} 
  binstate(const int& x) :x(x){}

  unsigned int return_L(){
    return L;
  }

  binstate& operator=(int x){
    this->x = x;
    return *this;
  }

  //* overloading operators for binstate x binstate
  binstate operator<<(binstate rhs){
    binstate result = *this;
    result.x = this->x << rhs.x;
    return result;
  }

  binstate operator>>(binstate rhs){
    binstate result = *this;
    result.x = this->x >> rhs.x;
    return result;
  }

  binstate operator^(binstate rhs){
    binstate result = *this;
    result.x = this->x ^ rhs.x;
    return result;
  }

  binstate operator&(binstate rhs){
    binstate result = *this;
    result.x = this->x & rhs.x;
    return result;
  }

  //* overloading operators for binstate x int
  binstate operator<<(int x){
    binstate result = *this;
    result.x = this->x << x;
    return result;
  }

  binstate operator>>(int x){
    binstate result = *this;
    result.x = this->x >> x;
    return result;
  }

  binstate operator^(int x){
    binstate result = *this;
    result.x = this->x ^ x;
    return result;
  }

  binstate operator&(int x){
    binstate result = *this;
    result.x = this->x & x;
    return result;
  }

  //* bitwise not operator
  binstate operator~(){
    binstate result = *this;
    result.x = ~this->x;
    return result;
  }

  // binstate& operator=(int x){
  //   this->x = x;
  //   return *this;
  // }

  // binstate& operator=(int x){
  //   this->x = x;
  //   return *this;
  // }

  friend ostream& operator<<(ostream& os, const binstate& al)
  {
    os << al.x;
    return os;
  }
};


int main (void)
{
  // int a = 3, b, c;
  // mpz_class d = 1;

  // d = d << a.get_ui();
  // cout << "d : " << d << endl;
  binstate<> y = 1, x = 2;
  cout << y.return_L() << endl; 




  //* state (using integer)
  int state_ = ~0;
  int lstate_ = ~0;
  auto t1 = high_resolution_clock::now();

  // long long int a,b,c;
  // long long int d;

  binstate<> a,b,c;
  for(std::size_t i=0; i<1E9; i++){
    
    a = 1;
    b = 20;
    c = a << b;
    c = c >> b;
  }
  // cout << x << endl;

  auto t2 = high_resolution_clock::now();
  auto elapsed = duration_cast<microseconds>(t2 - t1).count() / (double)1E3;

  cout << "elapsed time : " << elapsed << endl;
 
  return 0;
}