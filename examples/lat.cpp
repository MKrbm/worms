#include <iostream>
#include <vector>

class Line
{
public:
  Line(float begin, float end) : begin(begin), end(end) {}
  float GetLength(int x)
  {
    return (end - begin) * x;
  }

private:
  float begin;
  float end;
};

int main()
{

  float begin = 1.0;
  float end = 3.5;
  Line line(begin, end);

  std::cout << "Length of line: " << line.GetLength(3) << "\n";
  std::cout << "yeei" << std::endl;

  const std::vector<int> v = {1, 2, 3};

  // print the vector elements
  for (int i = 0; i < v.size(); i++)
    std::cout << v[i] << " ";

  // try to change the first element
  // this will cause a compile-time error
  v[0] = 4;

  return 0;
}