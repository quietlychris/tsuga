#![allow(non_snake_case)]

//! ## An early stage machine-learning library in Rust
//! `tsuga` is an early stage machine learning library in Rust for building neural networks. It uses `ndarray` as the linear algebra backend, and operates primarily on two-dimensional `f32` arrays (`Array2<f32>` types). At the moment, it's primary function has been for testing out various ideas for APIs, as an educational exercise, and probably isn't yet suitable for serious use. Most of the project's focus so far has been on the image-processing domain, although the tools  and layout should generally applicable to higher/lower-dimensional datasets as well.
//!
//! Tsuga currently uses the [Builder](https://xaeroxe.github.io/init-struct-pattern/) pattern for constructing fully-connected networks. Since networks are complex compound structures, this pattern helps to make the layout of the network explicit and modular.
//!
//! ### Dependencies
//! Tsuga uses the [`minifb`](https://github.com/emoon/rust_minifb) to display sample images during development, which means you may need to add certain dependencies via
//!
//! ```
//! $ sudo apt install libxkbcommon-dev libwayland-cursor0 libwayland-dev
//! ```
//!
//! ### MNIST Example
//! The following is a reduced-code example of building a network to train on/evaluate the MNIST (or Fashion MNIST) data set. Including unpacking the MNIST binary files, this network achieves:
//! - An accuracy of ~91.5% over 1000 iterations in 3.65 seconds  
//! - An accuracy of ~97.1% over 10,000 iterations in 29.43 seconds
//!
//! ```rust
//! use ndarray::prelude::*;
//! use tsuga::prelude::*;
//!
//! fn main() {
//!    // Reduced-version for importing the MNIST data and unpacking it into four Array2<f32> data structures
//!    let (input, output, test_input, test_output) = mnist_as_ndarray();
//!    println!("Successfully unpacked the MNIST dataset into Array2<f32> format!");
//!
//!    // Now we can begin configuring any additional hidden layers, specifying their size and activation function
//!    // We could also use activation functions like "relu"
//!    let mut layers_cfg: Vec<FCLayer> = Vec::new();
//!    let sigmoid_layer_0 = FCLayer::new("sigmoid", 128);
//!    layers_cfg.push(sigmoid_layer_0);
//!    let sigmoid_layer_1 = FCLayer::new("sigmoid", 64);
//!    layers_cfg.push(sigmoid_layer_1);
//!
//!    // The network can now be built using the specified layer configurations
//!    // Several other options for tuning the network's performance are available as well
//!    let mut fcn = FullyConnectedNetwork::default(input, output)
//!        .add_layers(layers_cfg)
//!        .iterations(1000)
//!        .learnrate(0.01)
//!        .batch_size(200)
//!        .build();
//!
//!    // Training occurs in place on the network
//!    fcn.train().expect("An error occurred while training");
//!
//!    // We can now pass an appropriately-sized input through our trained network,
//!    // receiving an Array2<f32> on the output
//!    let test_result = fcn.evaluate(test_input.clone());
//!
//!    // And will compare that output against the ideal one-hot encoded testing label array
//!    compare_results(test_result.clone(), test_output);
//!}

extern crate serial_test;

extern crate ndarray as nd;
extern crate ndarray_rand as ndr;
extern crate ndarray_stats as nds;

use nd::prelude::*;
use ndr::{rand_distr::Uniform, RandomExt};

/// Activation functions which can be applied element-wise or to subsets of the network's matrices
pub mod activation_functions;

/// Definitions for fully-connected layers which compose the neural networks
pub mod fc_layer;
/// Constructs, trains, and evaluates a neural network based on supplied input and output data
pub mod fc_network;

/// Definitions for convolutional layers
pub mod conv_layer;
/// An unstable and immature module for chaining static-kernel sliding-window convolutions of input data
pub mod conv_network;

/// Set of utility functions
pub mod utils;

mod test;

/// Contains all the necessary imports for building and training a basic neural network
pub mod prelude {
    pub use crate::activation_functions::*;

    pub use crate::fc_layer::*;
    pub use crate::fc_network::*;

    pub use crate::conv_layer::*;
    pub use crate::conv_network::*;

    pub use crate::utils::*;
}
