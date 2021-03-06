use crate::layer::Layer;
use ndarray::prelude::*;
use rand::Rng;
use std::error::Error;

pub struct Network {
    iterations: usize,
    input_shape: (usize, usize),
    input: Array4<f32>,
    output_shape: (usize, usize),
    output: Array2<f32>,
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn default(
        input_shape: (usize, usize),
        input: Array4<f32>,
        output_shape: (usize, usize),
        output: Array2<f32>,
    ) -> Self {
        Network {
            iterations: 25_000,
            input_shape,
            input,
            output_shape,
            output,
            layers: Vec::new(),
        }
    }

    pub fn set_iterations(&mut self, iterations: usize) {
        self.iterations = iterations;
    }

    pub fn add(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn train(&mut self) -> Result<(), Box<dyn Error>> {
        let mut rng = rand::thread_rng();
        for iteration in 0..self.iterations {
            let num = rng.gen_range(0..self.input.dim().0);
            let input = self
                .input
                .slice(s![num, .., .., ..])
                .into_shape(self.input_shape)
                .unwrap()
                .to_owned();

            let mut output = self.layers[0].forward(input)?;
            for i in 1..self.layers.len() {
                output = self.layers[i].forward(output)?;
            }

            let num_layers = self.layers.len() - 1;
            let actual = self
                .output
                .slice(s![num, ..])
                .into_shape(self.output_shape)
                .unwrap();
            let error = &output - &actual;
            if iteration % 1000 == 0 {
                println!("Error #{}: {}", iteration, error.sum());
            }

            let mut gradient = self.layers[num_layers].backward(error).unwrap();
            for i in { 0..self.layers.len() - 1 }.rev() {
                gradient = self.layers[i].backward(gradient)?;
            }
        }
        Ok(())
    }

    pub fn evaluate(&mut self, input: Array2<f32>) -> Result<Array2<f32>, Box<dyn Error>> {
        let mut output = self.layers[0].forward(input)?;
        for i in 1..self.layers.len() {
            output = self.layers[i].forward(output)?;
        }
        Ok(output)
    }

    pub fn info(&self) -> Result<(), Box<dyn Error>> {
        for layer in &self.layers {
            layer.print();
        }
        Ok(())
    }
}
