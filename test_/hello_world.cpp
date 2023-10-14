#include <mpi.h>
// #include <iil.h>
#include <iostream>

int main(int argc, char** argv) {
  MPI::Init(argc, argv);

  // Get the number of processes
  int world_size = MPI::COMM_WORLD.Get_size();

  // Get the rank of the process
  int world_rank = MPI::COMM_WORLD.Get_rank();

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI::Get_processor_name(processor_name,name_len);

  std::cout << "Hello World from processor " << processor_name << ", rank "
            << world_rank << " out of " << world_size << " processors\n";

  // Finalize the MPI environment.
  MPI::Finalize();
  std::cout << "Hello World from processor " << ", rank " << " out of " << " processors" << std::endl;
}
