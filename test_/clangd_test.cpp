#include <iostream>

using namespace std;
// Google's style guide prefers the use of nullptr over NULL.
// #define NULL 0

int main() {
    int* ptr = NULL;

    if (ptr == NULL) {
        std::cout << "Pointer is NULL" << std::endl;
    }

    return 0;
}
