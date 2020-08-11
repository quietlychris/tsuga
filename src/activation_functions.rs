use ndarray::prelude::*;

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn sigmoid_prime(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

#[inline]
pub fn relu(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

#[inline]
pub fn relu_prime(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else {
        1.
    }
}

#[inline]
pub fn threshold(x: f32, threshold: f32) -> f32 {
    if x > threshold {
        1.0
    } else {
        0.0
    }
}

#[inline]
pub fn softmax(array: &mut Array2<f32>) {
    for j in 0..array.nrows() {
        let mut sum = 0.;
        for i in 0..array.ncols() {
            sum += array[[j, i]];
        }
        for i in 0..array.ncols() {
            array[[j, i]] = array[[j, i]] / sum;
        }
    }
}
