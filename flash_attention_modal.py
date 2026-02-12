"""
FlashAttention-2 Implementation using Modal
FILE NAME: flash_attention_modal.py

Run with: modal run flash_attention_modal.py
"""

import modal

# Create Modal app
app = modal.App("flashattention-v2")

# Define image with build tools
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "wget")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-3"
    )
)

# C Implementation code
C_CODE = """
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

void matmul(float* A, float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int p = 0; p < n; p++) {
                sum += A[i * n + p] * B[p * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}

void rowmax(float* S, float* m, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float max_val = -FLT_MAX;
        for (int j = 0; j < cols; j++) {
            if (S[i * cols + j] > max_val) max_val = S[i * cols + j];
        }
        m[i] = max_val;
    }
}

void rowsum(float* P, float* l, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) sum += P[i * cols + j];
        l[i] = sum;
    }
}

void flash_attention_forward(float* Q, float* K, float* V, float* O, float* L,
                            int N, int d, int Br, int Bc) {
    int Tr = (N + Br - 1) / Br;
    int Tc = (N + Bc - 1) / Bc;
    float scale = 1.0f / sqrtf((float)d);
    
    for (int i = 0; i < Tr; i++) {
        int row_start = i * Br;
        int row_end = (row_start + Br < N) ? row_start + Br : N;
        int actual_Br = row_end - row_start;
        
        float* Qi = (float*)malloc(actual_Br * d * sizeof(float));
        float* Oi = (float*)calloc(actual_Br * d, sizeof(float));
        float* li = (float*)calloc(actual_Br, sizeof(float));
        float* mi = (float*)malloc(actual_Br * sizeof(float));
        
        for (int r = 0; r < actual_Br; r++) {
            mi[r] = -FLT_MAX;
            memcpy(&Qi[r * d], &Q[(row_start + r) * d], d * sizeof(float));
        }
        
        for (int j = 0; j < Tc; j++) {
            int col_start = j * Bc;
            int col_end = (col_start + Bc < N) ? col_start + Bc : N;
            int actual_Bc = col_end - col_start;
            
            float* Kj = (float*)malloc(actual_Bc * d * sizeof(float));
            float* Vj = (float*)malloc(actual_Bc * d * sizeof(float));
            float* Sij = (float*)malloc(actual_Br * actual_Bc * sizeof(float));
            float* Pij = (float*)malloc(actual_Br * actual_Bc * sizeof(float));
            float* mij = (float*)malloc(actual_Br * sizeof(float));
            float* lij = (float*)malloc(actual_Br * sizeof(float));
            
            for (int r = 0; r < actual_Bc; r++) {
                memcpy(&Kj[r * d], &K[(col_start + r) * d], d * sizeof(float));
                memcpy(&Vj[r * d], &V[(col_start + r) * d], d * sizeof(float));
            }
            
            matmul(Qi, Kj, Sij, actual_Br, d, actual_Bc);
            for (int idx = 0; idx < actual_Br * actual_Bc; idx++) Sij[idx] *= scale;
            
            rowmax(Sij, mij, actual_Br, actual_Bc);
            
            for (int r = 0; r < actual_Br; r++) {
                for (int c = 0; c < actual_Bc; c++) {
                    Pij[r * actual_Bc + c] = expf(Sij[r * actual_Bc + c] - mij[r]);
                }
            }
            
            rowsum(Pij, lij, actual_Br, actual_Bc);
            
            for (int r = 0; r < actual_Br; r++) {
                float mi_new = fmaxf(mi[r], mij[r]);
                float li_new = expf(mi[r] - mi_new) * li[r] + expf(mij[r] - mi_new) * lij[r];
                float correction = li[r] * expf(mi[r] - mi_new);
                
                for (int c = 0; c < d; c++) {
                    float pv_sum = 0.0f;
                    for (int k = 0; k < actual_Bc; k++) {
                        pv_sum += Pij[r * actual_Bc + k] * Vj[k * d + c];
                    }
                    Oi[r * d + c] = (Oi[r * d + c] * correction + pv_sum) / li_new;
                }
                
                mi[r] = mi_new;
                li[r] = li_new;
            }
            
            free(Kj); free(Vj); free(Sij); free(Pij); free(mij); free(lij);
        }
        
        for (int r = 0; r < actual_Br; r++) {
            memcpy(&O[(row_start + r) * d], &Oi[r * d], d * sizeof(float));
            L[row_start + r] = mi[r] + logf(li[r]);
        }
        
        free(Qi); free(Oi); free(li); free(mi);
    }
}

int main() {
    int N = 256, d = 64, Br = 32, Bc = 32;
    
    float* Q = (float*)malloc(N * d * sizeof(float));
    float* K = (float*)malloc(N * d * sizeof(float));
    float* V = (float*)malloc(N * d * sizeof(float));
    float* O = (float*)malloc(N * d * sizeof(float));
    float* L = (float*)malloc(N * sizeof(float));
    
    for (int i = 0; i < N * d; i++) {
        Q[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        K[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        V[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    printf("Running FlashAttention-2 (C Implementation)\\n");
    printf("N=%d, d=%d, Br=%d, Bc=%d\\n\\n", N, d, Br, Bc);
    
    flash_attention_forward(Q, K, V, O, L, N, d, Br, Bc);
    
    printf("✓ Completed successfully!\\n");
    printf("Sample outputs:\\n");
    printf("  O[0][0:5] = [%.4f, %.4f, %.4f, %.4f, %.4f]\\n", O[0], O[1], O[2], O[3], O[4]);
    printf("  L[0:5] = [%.4f, %.4f, %.4f, %.4f, %.4f]\\n", L[0], L[1], L[2], L[3], L[4]);
    
    free(Q); free(K); free(V); free(O); free(L);
    return 0;
}
"""

CUDA_CODE = """
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define CUDA_CHECK(call) do { \\
    cudaError_t err = call; \\
    if (err != cudaSuccess) { \\
        printf("CUDA Error: %s\\n", cudaGetErrorString(err)); \\
        exit(1); \\
    } \\
} while(0)

__global__ void flash_attention_kernel(
    const float* Q, const float* K, const float* V,
    float* O, float* L, int N, int d, int Br, int Bc
) {
    int block_row = blockIdx.x;
    int tid = threadIdx.x;
    int row_start = block_row * Br;
    int row_end = min(row_start + Br, N);
    int actual_Br = row_end - row_start;
    if (row_start >= N) return;
    
    float scale = rsqrtf((float)d);
    int Tc = (N + Bc - 1) / Bc;
    
    extern __shared__ float smem[];
    float* s_Qi = smem;
    float* s_Kj = s_Qi + Br * d;
    float* s_Vj = s_Kj + Bc * d;
    float* s_Sij = s_Vj + Bc * d;
    float* s_Pij = s_Sij + Br * Bc;
    float* s_Oi = s_Pij + Br * Bc;
    float* s_li = s_Oi + Br * d;
    float* s_mi = s_li + Br;
    float* s_lij = s_mi + Br;
    float* s_mij = s_lij + Br;
    
    for (int i = tid; i < actual_Br * d; i += blockDim.x) s_Oi[i] = 0.0f;
    for (int i = tid; i < actual_Br; i += blockDim.x) {
        s_li[i] = 0.0f;
        s_mi[i] = -FLT_MAX;
    }
    __syncthreads();
    
    for (int i = tid; i < actual_Br * d; i += blockDim.x) {
        s_Qi[i] = Q[(row_start + i / d) * d + i % d];
    }
    __syncthreads();
    
    for (int j = 0; j < Tc; j++) {
        int col_start = j * Bc;
        int col_end = min(col_start + Bc, N);
        int actual_Bc = col_end - col_start;
        
        for (int i = tid; i < actual_Bc * d; i += blockDim.x) {
            s_Kj[i] = K[(col_start + i / d) * d + i % d];
            s_Vj[i] = V[(col_start + i / d) * d + i % d];
        }
        __syncthreads();
        
        for (int i = tid; i < actual_Br * actual_Bc; i += blockDim.x) {
            int r = i / actual_Bc, c = i % actual_Bc;
            float sum = 0.0f;
            for (int k = 0; k < d; k++) sum += s_Qi[r * d + k] * s_Kj[c * d + k];
            s_Sij[i] = sum * scale;
        }
        __syncthreads();
        
        for (int r = tid; r < actual_Br; r += blockDim.x) {
            float max_val = -FLT_MAX;
            for (int c = 0; c < actual_Bc; c++) max_val = fmaxf(max_val, s_Sij[r * actual_Bc + c]);
            s_mij[r] = max_val;
        }
        __syncthreads();
        
        for (int i = tid; i < actual_Br * actual_Bc; i += blockDim.x) {
            s_Pij[i] = expf(s_Sij[i] - s_mij[i / actual_Bc]);
        }
        __syncthreads();
        
        for (int r = tid; r < actual_Br; r += blockDim.x) {
            float sum = 0.0f;
            for (int c = 0; c < actual_Bc; c++) sum += s_Pij[r * actual_Bc + c];
            s_lij[r] = sum;
        }
        __syncthreads();
        
        for (int r = tid; r < actual_Br; r += blockDim.x) {
            float mi_old = s_mi[r], mi_new = fmaxf(mi_old, s_mij[r]);
            float li_new = expf(mi_old - mi_new) * s_li[r] + expf(s_mij[r] - mi_new) * s_lij[r];
            float correction = (s_li[r] > 0.0f) ? expf(mi_old - mi_new) * s_li[r] / li_new : 0.0f;
            
            s_mi[r] = mi_new;
            for (int c = 0; c < d; c++) s_Oi[r * d + c] *= correction;
            s_li[r] = li_new;
        }
        __syncthreads();
        
        for (int i = tid; i < actual_Br * d; i += blockDim.x) {
            int r = i / d, c = i % d;
            float pv = 0.0f;
            for (int k = 0; k < actual_Bc; k++) pv += s_Pij[r * actual_Bc + k] * s_Vj[k * d + c];
            s_Oi[i] += pv / s_li[r];
        }
        __syncthreads();
    }
    
    for (int i = tid; i < actual_Br * d; i += blockDim.x) {
        O[(row_start + i / d) * d + i % d] = s_Oi[i];
    }
    for (int r = tid; r < actual_Br; r += blockDim.x) {
        L[row_start + r] = s_mi[r] + logf(s_li[r]);
    }
}

int main() {
    int N = 512, d = 64, Br = 32, Bc = 32;  // Reduced sizes for T4 GPU
    size_t mat_size = N * d * sizeof(float), vec_size = N * sizeof(float);
    
    float *h_Q = (float*)malloc(mat_size);
    float *h_K = (float*)malloc(mat_size);
    float *h_V = (float*)malloc(mat_size);
    float *h_O = (float*)malloc(mat_size);
    float *h_L = (float*)malloc(vec_size);
    
    for (int i = 0; i < N * d; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_K[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_V[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    CUDA_CHECK(cudaMalloc(&d_Q, mat_size));
    CUDA_CHECK(cudaMalloc(&d_K, mat_size));
    CUDA_CHECK(cudaMalloc(&d_V, mat_size));
    CUDA_CHECK(cudaMalloc(&d_O, mat_size));
    CUDA_CHECK(cudaMalloc(&d_L, vec_size));
    
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, mat_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, mat_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, mat_size, cudaMemcpyHostToDevice));
    
    int Tr = (N + Br - 1) / Br;
    size_t smem = (Br*d + Bc*d + Bc*d + Br*Bc + Br*Bc + Br*d + Br + Br + Br + Br) * sizeof(float);
    
    // Check shared memory limit
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("Running FlashAttention-2 (CUDA Implementation)\\n");
    printf("Device: %s\\n", prop.name);
    printf("N=%d, d=%d, Br=%d, Bc=%d\\n", N, d, Br, Bc);
    printf("Grid: %d blocks, Shared mem: %.2f KB (Limit: %.2f KB)\\n\\n", 
           Tr, smem/1024.0f, prop.sharedMemPerBlock/1024.0f);
    
    if (smem > prop.sharedMemPerBlock) {
        printf("ERROR: Shared memory required (%.2f KB) exceeds limit (%.2f KB)\\n", 
               smem/1024.0f, prop.sharedMemPerBlock/1024.0f);
        return 1;
    }
    
    flash_attention_kernel<<<Tr, 256, smem>>>(d_Q, d_K, d_V, d_O, d_L, N, d, Br, Bc);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_O, d_O, mat_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_L, d_L, vec_size, cudaMemcpyDeviceToHost));
    
    printf("✓ Completed successfully!\\n");
    printf("Sample outputs:\\n");
    printf("  O[0][0:5] = [%.4f, %.4f, %.4f, %.4f, %.4f]\\n", h_O[0], h_O[1], h_O[2], h_O[3], h_O[4]);
    printf("  L[0:5] = [%.4f, %.4f, %.4f, %.4f, %.4f]\\n", h_L[0], h_L[1], h_L[2], h_L[3], h_L[4]);
    
    free(h_Q); free(h_K); free(h_V); free(h_O); free(h_L);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_L);
    return 0;
}
"""

@app.function(image=image, timeout=600)
def run_c():
    import subprocess
    with open("/tmp/flash.c", "w") as f:
        f.write(C_CODE)
    subprocess.run(["gcc", "-o", "/tmp/flash_c", "/tmp/flash.c", "-lm", "-O3"], check=True)
    result = subprocess.run(["/tmp/flash_c"], capture_output=True, text=True)
    return result.stdout

@app.function(image=image, gpu="T4", timeout=600)
def run_cuda():
    import subprocess
    import os
    
    # Add CUDA to PATH
    cuda_path = "/usr/local/cuda-12.3/bin"
    os.environ["PATH"] = f"{cuda_path}:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = f"/usr/local/cuda-12.3/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
    
    with open("/tmp/flash.cu", "w") as f:
        f.write(CUDA_CODE)
    
    # Use full path to nvcc
    nvcc_path = "/usr/local/cuda-12.3/bin/nvcc"
    subprocess.run([nvcc_path, "-o", "/tmp/flash_cuda", "/tmp/flash.cu", "-arch=sm_75"], check=True)
    result = subprocess.run(["/tmp/flash_cuda"], capture_output=True, text=True)
    return result.stdout

@app.local_entrypoint()
def main():
    print("\n" + "="*70)
    print("FlashAttention-2 Algorithm 1 - Modal Implementation")
    print("="*70 + "\n")
    
    print("🔹 Part 1: C Implementation (CPU)")
    print("-"*70)
    print(run_c.remote())
    
    print("\n" + "="*70)
    print("🔹 Part 2: CUDA Implementation (GPU)")
    print("-"*70)
    print(run_cuda.remote())
    
    print("\n" + "="*70)
    print("✅ All implementations completed!")
    print("="*70)