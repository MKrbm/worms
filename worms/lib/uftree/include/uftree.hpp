#pragma once

#include <vector>
#include <iostream>
#include <cstddef>

class UnionFindTree {
private:


public:

  int N;           // 要素数
  std::vector<int> par;  // 各ノードの親のID（根ノードは自分を参照）
  std::vector<int> sizes;  // 各ノードを根とする木のサイズ（根でないノードには無関係）

  int check_index(int n);
  UnionFindTree(int n);
  int find(int x);
  void unite(int x, int y);
  bool same(int x, int y);
  void show();
  int size(int x);
  int get_N(){
    return N;
  }

  int add_group(int n = 1){

    #if DEBUG==1
    std::cout << "capacity before add new group : " << par.capacity() << std::endl;
    #endif
    for(int i=0;i<n;i++){
      par.push_back(N++);
      sizes.push_back(1);
    }
    #if DEBUG==1
    std::cout << "capacity after add new group : " << par.capacity() << std::endl;
    #endif
    return N;
  }
  virtual void initialize(){
    N = 0;
    par.resize(0);
    sizes.resize(0);
  }

};

