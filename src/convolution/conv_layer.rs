use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use std::error::Error;

use crate::convolution::hyperparameters::*;
use crate::layer::Layer;
use crate::prelude::*;

/// A fully-connected layer
#[derive(Debug)]
pub struct ConvLayer {
    pub input: Array3<f32>,
    pub hp: ConvHyperParam,
    pub w: Array2<f32>,
    pub z: Array2<f32>,
    pub a: Array2<f32>,
    pub delta: Array2<f32>,
    pub act_fn: String,
    pub learnrate: f32,
}

impl ConvLayer {
    fn new(hp: ConvHyperParam, learnrate: f32) -> Result<Box<Self>, Box<dyn Error>> {
        Ok(Box::new(ConvLayer {
            input: Default::default(),
            hp,
            act_fn: "sigmoid".to_string(),
            w: Default::default(),
            z: Default::default(),
            a: Default::default(),
            delta: Default::default(),
            learnrate,
        }))
    }
}

impl Layer for ConvLayer {
    fn print(&self) {}

    fn forward(&mut self, input: Array2<f32>) -> Result<Array2<f32>, Box<dyn Error>> {
        Ok(self.a.clone())
    }

    fn backward(&mut self, input_gradient: Array2<f32>) -> Result<Array2<f32>, Box<dyn Error>> {
        Ok(self.delta.clone())
    }
}
