#include <iostream>


class Base {
 public:

  virtual void print() const {
    std::cout << "Base\n";
  }
};

class Derived : public Base {
 public:
  int x;
  Derived(int x) : x(x), Base() {}
  void print() const override {
    std::cout << x << '\n';
  }
};

int main() {
  Base base(Derived(2));
  base.print();
  Base* derived_ptr = new Derived(2);
  Derived der = Derived(2);
  Base* derved_ptr2 = &der;
  Base& base_red = der;
  Base base_redef = der;
  Base base2(base_red);

  base2.print();
  base_red.print();
  derived_ptr->print();
  derved_ptr2->print();
  base_redef.print();
  // Base* base_ptr = new Base();
  // base_ptr->print();
  // derived_ptr->print();  // outputs "Base"

  return 0;
}

