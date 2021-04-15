use crate::layer::Layer;
use ndarray::prelude::*;
use std::error::Error;

#[derive(Debug)]
pub struct SigmoidLayer {
    pub z: Array2<f32>,
}

impl SigmoidLayer {
    pub fn new() -> Box<Self> {
        Box::new(SigmoidLayer {
            z: Default::default(),
        })
    }
}

impl Layer for SigmoidLayer {
    #[inline]
    fn forward(&mut self, input: Array2<f32>) -> Result<Array2<f32>, Box<dyn Error>> {
        self.z = input;
        Ok(self.z.mapv(|x| sigmoid(x)))
    }

    #[inline]
    fn backward(&mut self, input: Array2<f32>) -> Result<Array2<f32>, Box<dyn Error>> {
        // let delta = self.z.clone().mapv(|x| sigmoid_prime(x)) * input;
        Ok(self.z.clone().mapv(|x| sigmoid_prime(x)) * input)
    }

    fn print(&self) {
        println!("A sigmoid layer");
    }
}

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