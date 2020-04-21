use crate::conv_layer::*;
use ndarray::prelude::*;

#[derive(Debug, Clone)]
pub struct ConvolutionalNetwork {
    pub input: Array2<f64>,
    pub layers: Vec<ConvLayer>,
    outputs: Vec<Array2<f64>>,
}

impl ConvolutionalNetwork {
    pub fn default(input: Array2<f64>) -> Self {
        let default_kernel: Array2<f64> = array![[1., -1.], [-1., 1.]];
        ConvolutionalNetwork {
            input: input,
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
            input: self.input,
            layers: self.layers,
            outputs: self.outputs,
        }
    }

    pub fn network_convolve(&mut self) -> Array2<f64> {
        self.outputs.push(self.layers[0].convolve(&self.input));
        if self.layers.len() == 1 {
            self.outputs.last().unwrap().to_owned()
        } else {
            for i in 1..self.layers.len() {
                let output = self.layers[i].convolve(&self.outputs[i - 1]);
                self.outputs.push(output);
            }
            self.outputs.last().unwrap().to_owned()
        }
    }
}

#[test]
#[ignore]
fn basic_conv_network() {
    let input: Array2<f64> = array![[1., 2., 3., 4.], [4., 3., 2., 1.], [1., 2., 2.5, 4.]];
    let mut conv_layers: Vec<ConvLayer> = Vec::new();
    let conv_layer_0 = ConvLayer::default(array![[1., 0.], [0., 0.]])
        .kernel(array![[1., 0.], [0., 0.]])
        .build();
    let conv_layer_1 = ConvLayer::default(array![[1., 0.], [0., 0.]])
        .kernel(array![[1., 0.], [0., 1.]])
        .build();
    conv_layers.push(conv_layer_0);
    conv_layers.push(conv_layer_1);
    let mut conv_network: ConvolutionalNetwork = ConvolutionalNetwork::default(input)
        .add_layers(conv_layers)
        .build();
    println!("ConvNetwork:\n{:#?}", conv_network);
    let convolved_input = conv_network.network_convolve();
    println!("convolved_input:\n{:#?}", convolved_input);
}

#[test]
#[ignore]
fn basic_conv_layer_process() {
    let input: Array2<f64> = array![[1., 2., 3., 4.], [4., 3., 2., 1.], [1., 2., 2.5, 4.]];
    let (i_n, i_m) = (input.shape()[0], input.shape()[1]);
    println!("Input shape is: {:?}", (i_n, i_m));
    let kernel: Array2<f64> = array![[1., 0.], [0., 0.]];
    let (k_n, k_m) = (kernel.shape()[0], kernel.shape()[1]);
    println!("Kernel shape is: {:?}", (k_n, k_m));
    let (o_n, o_m) = (i_n - k_n + 1, i_m - k_m + 1);
    println!("Output shape is: {:?}", (o_n, o_m));

    let mut output: Array2<f64> = Array::zeros((o_n, o_m));
    println!("{:#?}", output);
    for y in 0..o_n {
        for x in 0..o_m {
            let input_subview = input.slice(s![y..(y + k_n), x..(x + k_m)]);
            // println!("input_subview:\n{:?}",input_subview);
            output[[y, x]] = (&input_subview * &kernel).sum();
        }
    }
    println!("Convolved matrix:\n{:#?}", output);
}
