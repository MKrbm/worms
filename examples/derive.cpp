#include <iostream>

// bool endsWith(const std::string &str, const std::string &suffix) {
//     return str.size() >= suffix.size() &&
//            str.rfind(suffix) == str.size() - suffix.size();
// }

// class Base {
//  public:

//   virtual void print() const {
//     std::cout << "Base\n";
//   }
//   void fun() 
//   {
//     std::cout << "Base fun\n";
//   }
// };

// class Derived : public Base {
//  public:
//   int x;
//   Derived(int x) : x(x), Base() {}
//   void print() const override {
//     std::cout << x << '\n';
//   }
// };

// int main() {
//   Base base(Derived(2));
//   Base* derived_ptr = new Derived(2);
//   Derived der = Derived(2);
//   Base base_redef = der;
//   Base& bder = *derived_ptr;

//   base.print();
//   base_redef.print();
//   derived_ptr->print();
//   bder.print();
//   bder.fun();
//   // Base* base_ptr = new Base();
//   // base_ptr->print();
//   // derived_ptr->print();  // outputs "Base"

//   // std::string str = "example.npy";
//   // std::string suffix = ".npy";

//   // std::cout << endsWith(str, suffix) << std::endl;
//   // std::cout << endsWith("example-npy", suffix) << std::endl;
//   // std::cout << endsWith("npy.npy", suffix) << std::endl;
//   // std::cout << endsWith("example..npy", suffix) << std::endl;
//   // std::cout << endsWith("foo/foo/foo.npy", suffix) << std::endl;



//   return 0;
// }

#include <iostream>

class Parent {
public:
    void setX(int x) { this->x = x; }
    int getX() const { return x; }

private:
    int x;
};

class Derived : public Parent {
public:
    void changeX() { Parent::setX(20); }
};

int main() {
    Derived d;
    d.changeX();
    std::cout << d.getX() << std::endl; // Outputs: 20

    return 0;
}