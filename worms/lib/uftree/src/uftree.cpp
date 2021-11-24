#include "../include/uftree.hpp"

#include <functional>
#include <iostream>

UnionFindTree::UnionFindTree(int n) : par(n), sizes(n, 1), N(n) {
  for (int i = 0; i < n; i++) par[i] = i;
}

int UnionFindTree::check_index(int g){
	// std::cout << "check if index is appropriate\n";
  if (g==-1){
    #if DEBUG==1
    std::cout << "node is not yet assigned" << std::endl;
    #endif
    return -1;
  }
	if (g < 0 || g >= N ){
		throw std::runtime_error("this index is inappropriate\n");
		return 1;
	}
	return 0;
}

// 要素xが属するグループの根ノードのIDを見つける
int UnionFindTree::find(int x) {
  if (check_index(x) == -1) return -1;
  if (par[x] == x)
      return x;
    else
      par[x] = find(par[x]);
      #if DEBUG==1
      std::cout << "\npar[" << x << "]" << " : " << par[x] << std::endl; 
      #endif
      return par[x];

}

// 要素x, yが属するグループ同士を統合する
void UnionFindTree::unite(int x, int y) {
  if ((x==-1) || (y ==-1)){
    #if DEBUG==1
    std::cout << "no node found " << std::endl;
    #endif
    return;
  }

  x = find(x);
  y = find(y);
  if (x == y) return;

  if (sizes[x] < sizes[y]) {
    par[x] = y;
    sizes[y] += sizes[x];
  } else {
    par[y] = x;
    sizes[x] += sizes[y];
  }
}



// グループのリストを表示する
// (IDが小さい順に表示される, O(n^2)なので最速の実装ではない)
void UnionFindTree::show() {
  std::cout << "Groups : " << std::endl;
  std::function<void(int)> f = [&](int x) {
    std::cout << x << ',';
    for (int y = 0; y < N; y++) {
      if (par[y] == x && y != x) f(y);
    }
  };
  for (int i = 0; i < N; i++) {
    if (par[i] == i) {
      f(i);
      std::cout << std::endl;
    }
  }
}

// 要素x, yが同じグループに属するかどうか
bool UnionFindTree::same(int x, int y) { 
	return find(x) == find(y); 
}

// 要素xが所属するグループに含まれる要素の数
int UnionFindTree::size(int x) { 
	return sizes[find(x)]; 
}