#include "../include/funcs.hpp"

void fillStringWithSpaces(std::string& str, int targetLength) {
  // Get the current length of the string
  int currentLength = str.length();
  // If the current length is less than the target length, append spaces
  if (currentLength < targetLength) {
    // Calculate how many spaces to append
    int spaces = targetLength - currentLength;
    // Append spaces to the end of the string
    str.append(spaces, ' ');
  }
}