// This is currently a 2D->2D flatten layer, since the Layer trait only uses
// Array2<f32> in the type signature. 
// TO_DO: That needs to be updated to allow for ArrayD for allowing intermixing between
// conv layer inputs and fully-connected layer inputs

use crate::layer::Layer;
use ndarray::prelude::*;
use std::error::Error;

#[derive(Debug)]
pub struct FlattenLayer {
    pub input: Array2<f32>
}

impl FlattenLayer {

    pub fn new() -> Box<Self> {
        Box::new(
            FlattenLayer {
                input: Default::default()
            }
        )
    }
}

impl Layer for FlattenLayer {

    fn forward(&mut self, input: Array2<f32>) -> Result<Array2<f32>, Box<dyn Error>> {
        self.input = input;
        let shape = self.input.shape();
        let output = self.input.clone().into_shape((1, shape[2] * shape[3]))?;
        Ok(output)
    }

    fn backward(&mut self, input: Array2<f32>) -> Result<Array2<f32>, Box<dyn Error>> {
        assert_eq!(input.shape(), self.input.shape());
        let shape = self.input.shape();
        let output = input.into_shape((shape[0], shape[1]))?;
        Ok(output)
    }

    fn print(&self) {
        println!("A flatten layer");
    }

}