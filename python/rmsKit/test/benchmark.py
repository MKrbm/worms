import torch
import time
import argparse



def generate_data(size, device, is_square=True):
    if is_square:
        return torch.randn(size, size, device=device), torch.randn(size, size, device=device)
    else:
        return torch.randn(size, device=device)


def run_benchmark(operation, device, data_size, print_grads=False):
    # Generate tensors with requires_grad=True to track gradients
    A, B = generate_data(data_size, device)
    A.requires_grad = True
    if B is not None:
        B.requires_grad = True

    start_time = time.time()

    # Forward pass
    result = operation(A, B)

    # Ensure GPU operations are completed if running on GPU
    if device == 'cuda':
        torch.cuda.synchronize()

    # Backward pass (if applicable)
    if result.requires_grad:
        result.sum().backward()
        if device == 'cuda':
            torch.cuda.synchronize()  # Synchronize after backward pass

    end_time = time.time()

    # Print gradients if required
    # if print_grads:
    #     if A.grad is not None:
    #         print(f"Gradient w.r.t. A:\n{A.grad.sum()}")
    #     if B.grad is not None:
    #         print(f"Gradient w.r.t. B:\n{B.grad.sum()}")

    return end_time - start_time


def kron_operation(A, B):
    return torch.kron(A, B)


def eigvalsh_operation(A, _):
    A = A + A.T
    return torch.linalg.eigvalsh(A)


def matrix_mult_operation(A, B):
    return torch.matmul(A, B)


# def benchmark_on_device(device, operations):
#     print(f"\nBenchmarking on {device.upper()}")
#     for operation, name, data_size in operations:
#         if device == 'cuda' and not torch.cuda.is_available():
#             print(f"{name}: CUDA not available")
#             continue
#         if device == 'mps' and not torch.backends.mps.is_available():
#             print(f"{name}: MPS not available")
#             continue
#
#         torch.cuda.empty_cache()  # Clear cache if on GPU
#         torch.manual_seed(0)  # For reproducibility
#         exec_time = run_benchmark(operation, device, data_size)
#         print(f"{name} ({data_size}{'x' + str(data_size) if operation != gram_schmidt_qr else ''}): {exec_time:.4f} seconds")


def benchmark_on_device(device, operations, repetitions):
    print(f"\nBenchmarking on {device.upper()}")
    for operation, name, data_size in operations:
        times = []

        for _ in range(repetitions):
            if device == 'cuda' and not torch.cuda.is_available():
                # Infinitely large time if CUDA not available
                times.append(float('inf'))
                break
            if device == 'mps' and not torch.backends.mps.is_available():
                # Infinitely large time if MPS not available
                times.append(float('inf'))
                break

            torch.cuda.empty_cache()  # Clear cache if on GPU
            torch.manual_seed(_)  # For reproducibility
            exec_time = run_benchmark(
                operation, device, data_size, print_grads=True)
            times.append(exec_time)

        avg_time = sum(times) / len(times)
        print(f"{name} ({data_size}{'x' + str(data_size)}): {avg_time:.4f} seconds (avg over {repetitions} runs)")

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Benchmarking with specified number of CPUs.')
    parser.add_argument('-n', '--num-cpus', type=int, default=torch.get_num_threads(),
                        help='Number of CPUs to use')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    # Set the number of threads in PyTorch
    torch.set_num_threads(args.num_cpus)
    num_threads = torch.get_num_threads()
    print(f"Number of threads set to: {num_threads}")
    operations = [
        (kron_operation, "Kronecker Product", 100),  # Adjust data_size as needed
        (eigvalsh_operation, "Eigenvalues", 2000),
        (matrix_mult_operation, "Matrix Multiplication", 5000),
    ]
    num_threads = torch.get_num_threads()
    print(f"Number of threads: {num_threads}")

    for device in ["cpu", "cuda"]:
        benchmark_on_device(device, operations, 10)
