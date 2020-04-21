#![allow(non_snake_case)]

extern crate ndarray as nd;
extern crate ndarray_linalg as ndl;
extern crate ndarray_rand as ndr;
extern crate ndarray_stats as nds;

use nd::prelude::*;
use ndr::{rand_distr::Uniform, RandomExt};

pub mod activation_functions;

pub mod fc_layer;
pub mod fc_model;
pub mod fc_network;

pub mod conv_layer;
pub mod conv_network;

mod test;

pub mod prelude {
    pub use crate::activation_functions::*;

    pub use crate::fc_layer::*;
    pub use crate::fc_model::*;
    pub use crate::fc_network::*;

    pub use crate::conv_layer::*;
    pub use crate::conv_network::*;
}
