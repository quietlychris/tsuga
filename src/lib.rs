//! ## An early stage machine-learning library in Rust
//! Tsuga is an early-stage machine learning library in Rust for building neural networks.
//! It uses [`ndarray`](https://github.com/rust-ndarray/ndarray) as the linear algebra backend.
//! At the moment, it's primary function has been for testing out various ideas for APIs, as
//! an educational exercise, and probably isn't yet suitable for serious use. Most of the
//! project's focus so far has been on the image-processing domain, although the tools  
//! and layout should generally applicable to higher/lower-dimensional datasets as well.

/// Activation functions which can be applied element-wise or to subsets of the network's matrices
pub mod activation_functions;
/// Fully-connected layers
pub mod fully_connected;
/// Definition for the Layer Trait
pub mod layer;
/// Network structure which holds multiple Layers
pub mod network;

/// Contains all the necessary imports for building and training a basic neural network
pub mod prelude {
    pub use crate::activation_functions::*;
    pub use crate::fully_connected::*;
    pub use crate::layer::*;
    pub use crate::network::*;
}
