import torch
import time

def eigvalsh_cpu(matrix):
    start = time.time()
    eigenvalues = torch.linalg.eigvalsh(matrix)
    end = time.time()
    return eigenvalues, end - start

def eigvalsh_gpu(matrix):
    start = time.time()
    eigenvalues = torch.linalg.eigvalsh(matrix.cuda())
    end = time.time()
    return eigenvalues.cpu(), end - start

# Set the matrix size and number of iterations
matrix_size = 2000
num_iterations = 100

# Create a random symmetric matrix
matrix = torch.randn(matrix_size, matrix_size)
matrix = matrix + matrix.T

# Perform eigvalsh on CPU
cpu_eigenvalues = []
cpu_times = []
for _ in range(num_iterations):
    eigenvalues, execution_time = eigvalsh_cpu(matrix)
    cpu_eigenvalues.append(eigenvalues)
    cpu_times.append(execution_time)

# Perform eigvalsh on GPU
gpu_eigenvalues = []
gpu_times = []
for _ in range(num_iterations):
    eigenvalues, execution_time = eigvalsh_gpu(matrix)
    gpu_eigenvalues.append(eigenvalues)
    gpu_times.append(execution_time)

# Calculate average execution times
avg_cpu_time = sum(cpu_times) / num_iterations
avg_gpu_time = sum(gpu_times) / num_iterations

print(f"Average execution time on CPU: {avg_cpu_time:.5f} seconds")
print(f"Average execution time on GPU: {avg_gpu_time:.5f} seconds")

# Compare the eigenvalues computed on CPU and GPU
if all(torch.allclose(cpu_eigenvalues[i], gpu_eigenvalues[i]) for i in range(num_iterations)):
    print("Eigenvalues computed on CPU and GPU are equal.")
else:
    print("Eigenvalues computed on CPU and GPU are different.")

print("-" * 80)

def eigvalsh_cpu(matrix):
    matrix1 = matrix.clone()
    with torch.autograd.profiler.profile(use_cuda=False, record_shapes=True) as prof:
        eigenvalues = torch.linalg.eigvalsh(matrix1).sum()
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    return eigenvalues

def eigvalsh_gpu(matrix):
    matrix1 = matrix.clone().to(device = "cuda")
    with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True) as prof:
        eigenvalues = torch.linalg.eigvalsh(matrix1).sum()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    return eigenvalues.cpu()


# Perform eigvalsh on CPU
print("CPU Profiler Output:")
cpu_eigenvalues = eigvalsh_cpu(matrix)

# Perform eigvalsh on GPU
print("\nGPU Profiler Output:")
gpu_eigenvalues = eigvalsh_gpu(matrix)

# Compare the eigenvalues computed on CPU and GPU
if torch.allclose(cpu_eigenvalues, gpu_eigenvalues):
    print("\nEigenvalues computed on CPU and GPU are equal.")
else:
    print("\nEigenvalues computed on CPU and GPU are different.")