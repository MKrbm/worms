#include <iostream>

using namespace std;
int main()
{
    size_t a = 4, b = 0;
    int c = 3;
    std::cout << ((b - c) % a) << std::endl;
    std::cout << (b - c) << std::endl;

    if (c == 3) goto label;

    cout << "Before the label." << endl;


    label:
      cout << "Hi!  I'm a label!" << endl;
      cout <<"I'm a label too!" << endl;
    
    cout << "I'm not a label!" << endl;

    int x = 0;
    int const * p = &x;
    int y = 0;
    p = &y;
    cout << *p << endl;
    return 0;
}