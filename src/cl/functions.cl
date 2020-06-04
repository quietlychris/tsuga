// Much of this code got pulled from the work on `gpuarray-rs` by tedsta
// at https://github.com/tedsta/gpuarray-rs
// and is licensed under the MIT License 


// ADD SCALAR
__kernel void add(__global float* buffer, float scalar) {
    buffer[get_global_id(0)] += scalar; 
}

// HADAMARD/ARRAY ELEMENT-WISE MULTIPLICATION
__kernel void hadamard(__global const float *a,
                       __global const float *b,
                                  __global float *c) {
    uint i = get_global_id(0);
    c[i] = a[i] * b[i];
}

// DOT PRODUCT
__kernel void matmul(__global const float *a,
                     __global const float *b,
                           __global float *c,
                               const ulong M,
                               const ulong K)
{
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);

    float val = 0.0;
    for (ulong k = 0; k < M; k++) {
        val += a[i*M + k] * b[k*K + j];
    }
    c[i*K + j] = val;
}

// MULTIPLY BY SCALAR
__kernel void multiply_by_scalar(
            __private float const coeff,
            __global float const* const src,
            __global float* const res)
{
    uint const idx = get_global_id(0);
    res[idx] = src[idx] * coeff;
}

// SIGMOID
float sigmoid_op(float z){return 1.0/(1.0+exp(-z));}
__kernel void sigmoid(__global const float *a,
                                __global float *b) {
    uintptr_t i = get_global_id(0);
    b[i] = sigmoid_op(a[i]);
}


__kernel void transpose(__global const float *a,
                                   __global float *b,
                                   const ulong rows,
                                   const ulong cols) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    b[j*rows + i] = a[i*cols + j]; // Flip the dimensions
}

