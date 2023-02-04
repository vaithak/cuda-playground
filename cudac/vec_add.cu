#include <stdio.h>

__global__ void vec_add(float *a, float *b, float *res, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        res[i] = a[i] + b[i];
    }
}

int main() {
    int n = 100;

    // allocate memory on the host.
    float *a, *b, *res;
    a = (float *)malloc(n * sizeof(float));
    b = (float *)malloc(n * sizeof(float));
    res = (float *)malloc(n * sizeof(float));

    // initialize the host arrays.
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }

    // allocate memory on the device.
    float *d_a, *d_b, *d_res;
    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));
    cudaMalloc((void**)&d_res, n * sizeof(float));

    // copy the host arrays to the device.
    cudaMemcpy((void*)d_a, (void*)a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_b, (void*)b, n * sizeof(float), cudaMemcpyHostToDevice);

    // launch the kernel.
    vec_add<<<(n + 255) / 256, 256>>>(d_a, d_b, d_res, n);

    // copy the result back to the host.
    cudaMemcpy((void*)res, (void*)d_res, n * sizeof(float), cudaMemcpyDeviceToHost);

    // print the first and last 10 elements of the result.
    printf("First 10 elements:\n");
    for (int i = 0; i < 10; i++) {
        printf("\t%f + %f = %f\n", a[i], b[i], res[i]);
    }
    printf("...\n");
    printf("Last 10 elements:\n");
    for (int i = 0; i < 10; i++) {
        printf("\t%f + %f = %f\n", a[n-10+i], b[n-10+i], res[n-10+i]);
    }

    // free the memory.
    free(a);
    free(b);
    free(res);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    return 0;
}
