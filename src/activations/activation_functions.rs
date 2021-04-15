use ndarray::prelude::*;

/// REctified Linear Unit function
#[inline]
pub fn relu(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

/// Derivative of the ReLU function
#[inline]
pub fn relu_prime(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else {
        1.
    }
}

/// Applies the softmax function in-place on a mutable two-dimensional array, making sure that every row has a proportional value that sums to 1.0
#[inline]
pub fn softmax(array: &mut Array2<f32>) {
    for j in 0..array.nrows() {
        let mut sum = 0.;
        for i in 0..array.ncols() {
            sum += array[[j, i]];
        }
        for i in 0..array.ncols() {
            array[[j, i]] /= sum;
        }
    }
}
