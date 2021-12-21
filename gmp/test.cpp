#include <iostream>
#include <fstream>
#include <stdio.h>
#include <chrono>
#include <memory>
#include <iterator>
#include <cstdio>
#include <gmpxx.h>


using namespace std::chrono;
using std::cout;
using std::endl;
using std::ofstream;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
using std::chrono::microseconds;


mpz_class add(mpz_class a, mpz_class b){
  return a+b;
}

inline mpz_class modifyBit(mpz_class n, mpz_class p, mpz_class b)
{
    return ((n & ~(1 << p.get_ui())) | (b << p.get_ui()));
}

inline mpz_class getbit(mpz_class n, mpz_class p)
{
    return (n >> p.get_ui()) ^ 1;
}


int main (void)
{
  // int a = 3, b, c;
  // mpz_class d = 1;

  // d = d << a.get_ui();
  // cout << "d : " << d << endl;

  //* state (using integer)
  mpz_class state_ = ~0;
  mpz_class lstate_ = ~0;
  auto t1 = high_resolution_clock::now();

  long long int a,b,c;
  long long int d;
  double x = 0;
  for(std::size_t i=0; i<1E9; i++){
    
    a = 1;
    b = 20;
    c = a << b;
    c = c >> b;
    x++;
    // d += c;
    // if (i % (int)1E7 == 0) cout << c << endl;
    // int s1 = (i*100)%6;
    // int s2 = (i*99)%6;
    // int s3 = (i*98)%6;
    // int s4 = (i*97)%6;
    // int s5 = (i*96)%6;

    // state_ = state_ ^ (1<<s1);
    // state_ = modifyBit(s2, state_, 1);
    // state_ = modifyBit(s3, state_, 0);
    // lstate_ = modifyBit(0, lstate_, getbit(state_, s4)); 
    // lstate_ = modifyBit(1, lstate_, getbit(state_, s5)); 
  }
  cout << x << endl;

  auto t2 = high_resolution_clock::now();
  auto elapsed = duration_cast<microseconds>(t2 - t1).count() / (double)1E3;

  cout << "elapsed time : " << elapsed << endl;
 
  return 0;
}