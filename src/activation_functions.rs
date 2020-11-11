use ndarray::prelude::*;

/// Applies the sigmoid logistic function
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivative of the sigmoid function
#[inline]
pub fn sigmoid_prime(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

/// REctified Linear Unit function
#[inline]
pub fn relu(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

/// Hyperbolic tangent function
#[inline]
pub fn tanh(x: f32) -> f32 {
    x.tanh()
}

/// Derivative of Hyperbolic tangent function
#[inline]
pub fn tanh_prime(x: f32) -> f32 {
    1.0f32 - (x.tanh() * x.tanh())
}

//// Linear or Indentiy function
#[inline]
pub fn linear(x: f32) -> f32 {
    x
}

////  Derivative of linear or Indentiy function
#[inline]
pub fn linear_prime(_: f32) -> f32 {
    1.
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
            array[[j, i]] = array[[j, i]] / sum;
        }
    }
}
