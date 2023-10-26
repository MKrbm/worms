#include "../include/funcs.hpp"
#include <regex>

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




// void print_call_stack() {
//     const size_t max_depth = 100;
//     void* callstack[max_depth];
//     int stack_depth = backtrace(callstack, max_depth);
//
//     char** symbols = backtrace_symbols(callstack, stack_depth);
//     std::regex address_regex("\\[(.*)\\]");
//
//     for (int i = 1; i < stack_depth; ++i) {
//         std::string symbol(symbols[i]);
//         std::smatch match;
//
//         if (std::regex_search(symbol, match, address_regex) && match.size() > 1) {
//             std::string address = match[1].str();
//             std::cout << "  " << address << std::endl;
//
//             std::string cmd = "addr2line -e ";
//             cmd += program_invocation_name;
//             cmd += " -f -C -p ";
//             cmd += address;
//
//             std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
//             if (!pipe) {
//                 std::cerr << "popen() failed!" << std::endl;
//                 continue;
//             }
//
//             char buffer[128];
//             while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
//                 std::cout << "  " << buffer;
//             }
//         }
//     }
//
//     free(symbols);
// }
