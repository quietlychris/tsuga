use ndarray::prelude::*;
use std::error::Error;

pub trait Layer {
    fn forward(&mut self, input: Array2<f32>) -> Result<Array2<f32>, Box<dyn Error>>;
    fn backward(&mut self, input_gradient: Array2<f32>) -> Result<Array2<f32>, Box<dyn Error>>;
    fn print(&self);
}
