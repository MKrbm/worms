import torch

n = 16

def kron_operation(device):
    H = torch.randn(n*n, n*n, dtype=torch.complex64, requires_grad=False).to(device)
    H = H + H.T

    u = torch.randn(n, n, dtype=torch.complex64, requires_grad=True, device=device)
    U = torch.kron(u, u)
    H1 = torch.mm(U, torch.mm(H, U.H))
    H1 = torch.abs(H1)
    res = torch.linalg.eigvalsh(H1).sum().item()
    print(res)

# CPU operation
print("CPU Operation:")
device = torch.device("cpu")
kron_operation(device)  # Warm-up the CPU

with torch.autograd.profiler.profile(use_cuda=False) as prof:
    for _ in range(10):
        kron_operation(device)
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))


# GPU operation
if torch.cuda.is_available():
    print("\nGPU Operation:")
    device = torch.device("cuda")
    kron_operation(device)  # Warm-up the GPU

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for _ in range(10):
            kron_operation(device)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
else:
    print("CUDA is not available on this device.")