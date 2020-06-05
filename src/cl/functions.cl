// Much of this code got pulled from the work on `gpuarray-rs` by tedsta
// at https://github.com/tedsta/gpuarray-rs
// and is licensed under the MIT License 


// ADD SCALAR
__kernel void add_scalar(__global float* buffer, float scalar) {
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
__kernel void dot_product(__global const float* A, 
                          __global const float* B,
                          __global float* C,
                          ulong M,    
                          ulong K ) {
  
  ulong row = get_global_id(0);
  ulong column = get_global_id(1);

  float sum = 0.0;
  for (ulong i = 0; i < M; i++) {
    sum += A[row * M + i] * B[i * K + column];
  }
  C[row * K + column] = sum;
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

// ADDITION OF TWO SAME-SIZE VECTORS
__kernel void add(__global const float *a,
                       __global const float *b,
                                  __global float *c) {
    uint i = get_global_id(0);
    c[i] = a[i] + b[i];
}

// SUBTRACTION OF TWO SAME-SIZE VECTORS
__kernel void subtract(__global const float *a,
                       __global const float *b,
                             __global float *c) {
    uint i = get_global_id(0);
    c[i] = a[i] - b[i];
}