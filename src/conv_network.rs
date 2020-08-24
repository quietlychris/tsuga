use crate::conv_layer::*;
use ndarray::prelude::*;

/// Preliminary form of a network of simple convolution operations chained on an input
#[derive(Debug, Clone)]
pub struct ConvolutionalNetwork {
    pub layers: Vec<ConvLayer>,
    outputs: Vec<Array2<f32>>,
}

impl ConvolutionalNetwork {
    /// Builds a convolutional network structure
    pub fn default() -> Self {
        let default_kernel: Array2<f32> = array![[1., -1.], [-1., 1.]];
        ConvolutionalNetwork {
            layers: vec![ConvLayer::default(&default_kernel)],
            outputs: vec![],
        }
    }

    /// Adds layers from a Vec<ConvLayer> configuration to the network's architecture
    pub fn add_layers(mut self, layers: Vec<ConvLayer>) -> Self {
        self.layers = layers;
        self
    }

    /// Returns the final architecture of the Convolutional Network
    pub fn build(self) -> ConvolutionalNetwork {
        ConvolutionalNetwork {
            layers: self.layers,
            outputs: self.outputs,
        }
    }

    /// Runs a set of chained convolutions on the network's inputs, and returns the result
    pub fn network_convolve(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.outputs.push(self.layers[0].convolve(input));
        if self.layers.len() > 1 {
            for i in 1..self.layers.len() {
                let output = self.layers[i].convolve(&self.outputs[i - 1]);
                self.outputs.push(output.clone());
            }
        }

        self.outputs.last().unwrap().to_owned()
    }
}
