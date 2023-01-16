#include <stdio.h>
#include <mpi.h>
#include <iostream>
#include <vector>
using namespace std;

class MyData {
  private:
    int inner_value;
    float float_test;
    // vector<double> vec;
  public:
    int value;
    MyData(int v, float f, int n):inner_value(v), float_test(f){}
    MyData(const MyData& my){
      cout << "hello" << endl;
      inner_value =  my.inner_value;
      float_test = my.float_test;
      // vec = my.vec;
    }
    void print_value(int v = 1, int rank = 0){
      value = v;
      cout << "rank : " << rank << endl;
      cout << "value : " << value << "\t and memory address : " << reinterpret_cast<long>(&value) << endl;
      cout << "float_test : " << float_test << "\t and memory address : " << reinterpret_cast<long>(&float_test) << endl;
      // cout << "size : "<< vec.size() <<  " vector pointer : " << &vec << endl;
      // cout << value << inner_value <<  endl;
      // cout << float_test << endl;
    }
};

int main(int argc, char** argv){
    int rank, size;
    MyData mydata(0, 0, 4);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        mydata = MyData(3, 4, 6);
    }
    MPI_Bcast(&mydata, sizeof(MyData), MPI_BYTE, 0, MPI_COMM_WORLD);
    printf("Hello World from process %d of %d\n", rank, size);
    mydata.print_value(1, rank);
    MPI_Finalize();

    return 0;
}