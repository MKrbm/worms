#include <iostream>

using namespace std;

class Base {
public:
  int a;
  virtual void func() {
    cout << "foo" << endl;
  }
};

class Child: public Base {
public:
  int b;
  void func() {
    cout << "foo child" << endl;
  }
};

class Child2: public Base {
  public:
  int b;
  void func() {
    cout << "foo child2" << endl;
  }
};

int main() {
  Base *baseptr[2] = {new Child(), new Child2()};
  // cout << dynamic_cast<Child *>(baseptr[0])->func() << endl;
  cout << baseptr[0]->a << endl;
  baseptr[0]->func();
  baseptr[1]->func();
  Child *c = new Child();
  c->func();
  return 0;
}