import time
import torch
import triton
import triton.language as tl


def vector_add_pseudocode(a,b,output,grid):
    PIDS = torch.arange(grid[0])
    a_ptr = 0
    b_ptr = 0
    output_ptr = 0
    for pid in PIDS:
        a_ptr_offset = a_ptr + pid
        b_ptr_offset = b_ptr + pid
        output_ptr_offset = output_ptr + pid

        output[output_ptr_offset] = a[a_ptr_offset] + b[b_ptr_offset]

def vector_add_pseudocode_wrapper(a,b):
    n_elements = len(a)
    grid = (n_elements,)
    output = torch.empty_like(a)
    vector_add_pseudocode(a,b,output,grid)
    return output

@triton.jit
def vector_add(a_ptr,b_ptr,output_ptr):
    pid = tl.program_ids(axis = 0)
    a_ptr_offset = a_ptr + pid
    b_ptr_offset = b_ptr + pid
    output_ptr_offset = output_ptr + pid

    a = tl.load(a_ptr_offset)
    b = tl.load(b_ptr_offset)

    output = a+b

    tl.store(output_ptr_offset,output)

def vector_add_wrapper(a,b):
    output = torch.empty_like(a)
    n_elements = len(a)
    grid = (n_elements, )

    vector_add[grid](a,b,output)
    return output

def benchmark_torch(a,b,iters=100):
    for _ in range(10):
        _ = a+b
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        out = a+b
    torch.cuda.synchronize()
    end = time.time()

    return (end-start)/iters


def benchmark_triton(a,b,iters=100):
    for _ in range(10):
        _ = vector_add_wrapper(a,b)

    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        out = vector_add_wrapper(a,b)
    torch.cuda.synchronize()
    end = time.time()

    return (end-start)/iters

if __name__=="__main__":
    a = torch.randn(1000).to("cuda")
    b = torch.randn(1000).to("cuda")

    pseudo_output = vector_add_pseudocode_wrapper(a,b)
    triton_output = vector_add_wrapper(a,b)
    torch_output = a + b

    assert  torch.allclose(pseudo_output,triton_output)
    assert  torch.allclose(triton_output,torch_output)

    print("Triton Matches Torch!")
    N = 50_000_000
    a = torch.randn(N,device="cuda")
    b = torch.randn(N,device="cuda")

    torch_time = benchmark_torch(a,b)
    triton_time = benchmark_triton(a,b)

    print(f"Torch time per iter: {torch_time * 1e3:.3f} ms")
    print(f"Triton time per iter: {triton_time * 1e3:.3f} ms")

