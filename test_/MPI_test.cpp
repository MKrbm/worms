#include <stdio.h>
#include <mpi.h>
#include <iostream>
#include <vector>
using namespace std;

int main(int argc, char** argv){
    // int process_Rank, size_Of_Cluster;

    // MPI_Init(&argc, &argv);
    // MPI_Comm_size(MPI_COMM_WORLD, &size_Of_Cluster);
    // MPI_Comm_rank(MPI_COMM_WORLD, &process_Rank);
    
    // std::cout << "Is this a c++ ?" << std::endl;
    // printf("Hello World from process %d of %d\n", process_Rank, size_Of_Cluster);
    // MPI_Finalize();

    // int rank, size;
    // MPI_Init(&argc, &argv);
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &size);

    // int n = 100;
    // int* data = new int[n];
    // if (rank == 0) {
    //     for (int i = 0; i < n; i++) {
    //         data[i] = i;
    //     }
    // }

    // int local_n = n / size;
    // int* local_data = new int[local_n];
    // MPI_Scatter(data, local_n, MPI_INT, local_data, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // int local_sum = 0;
    // for (int i = 0; i < local_n; i++) {
    //     local_sum += local_data[i];
    // }

    // cout << local_sum << endl;
    // int global_sum;
    // MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // if (rank == 0) {
    //     cout << "The global sum is: " << global_sum << endl;
    // }

    // MPI_Finalize();


    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 100;
    vector<int> data;
    if (rank == 0) {
        data.resize(n);
        for (int i = 0; i < n; i++) {
            data[i] = i;
        }
    }

    int local_n = n / size;
    vector<int> local_data(local_n);
    MPI_Scatter(data.data(), local_n, MPI_INT, local_data.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

    int local_sum = 0;
    for (int i = 0; i < local_n; i++) {
        local_sum += local_data[i];
    }

    int global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "The global sum is: " << global_sum << endl;
    }

    MPI_Finalize();
    return 0;
}