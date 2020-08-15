use crate::conv_layer::*;
use ndarray::prelude::*;

#[derive(Debug, Clone)]
pub struct ConvolutionalNetwork {
    pub layers: Vec<ConvLayer>,
    outputs: Vec<Array2<f32>>,
}

impl ConvolutionalNetwork {
    pub fn default() -> Self {
        let default_kernel: Array2<f32> = array![[1., -1.], [-1., 1.]];
        ConvolutionalNetwork {
            layers: vec![ConvLayer::default(&default_kernel)],
            outputs: vec![],
        }
    }

    pub fn add_layers(mut self, layers: Vec<ConvLayer>) -> Self {
        self.layers = layers;
        self
    }

    pub fn build(self) -> ConvolutionalNetwork {
        ConvolutionalNetwork {
            layers: self.layers,
            outputs: self.outputs,
        }
    }

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
