#pragma once
#ifndef NDEBUG
#define printd(fmt, ...) printf(fmt, ##__VA_ARGS__)
#define dout cout 
#define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#define ASSERT(condition, message) do { } while (false)
#define printd(fmt, ...) 0
#define dout 0 && cout
#endif

#include <iostream>
#include <vector>
#include <execinfo.h>
#include <cstdlib>
#include <cxxabi.h>
#include <memory>
#include <cstdio>

template<class T> std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[ ";
    for ( const T& item : vec )
        os << item << ", ";
    os << "]"; return os;
}

void fillStringWithSpaces(std::string& str, int targetLength);

void print_call_stack();