#include <vector>


template <typename T>
std::vector<T> make_vector(std::size_t size) {
    return std::vector<T>(size);
}

template <typename T, typename... Args>
auto make_vector(std::size_t first, Args... sizes){
    auto inner = make_vector<T>(sizes...);
    return std::vector<decltype(inner)>(first, inner);
}