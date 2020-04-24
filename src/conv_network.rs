use crate::conv_layer::*;
use image::*;
use ndarray::prelude::*;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ConvolutionalNetwork {
    pub layers: Vec<ConvLayer>,
    outputs: Vec<Array2<f64>>,
    write_intermediate_results: (bool, String),
}

impl ConvolutionalNetwork {
    pub fn default() -> Self {
        let default_kernel: Array2<f64> = array![[1., -1.], [-1., 1.]];
        ConvolutionalNetwork {
            layers: vec![ConvLayer::default(&default_kernel)],
            outputs: vec![],
            write_intermediate_results: (false, "data/results/".to_string()),
        }
    }

    pub fn add_layers(mut self, layers: Vec<ConvLayer>) -> Self {
        self.layers = layers;
        self
    }

    pub fn write_intermediate_results(
        mut self,
        write_intermediate_results: (bool, String),
    ) -> Self {
        self.write_intermediate_results = write_intermediate_results;
        self
    }

    pub fn build(self) -> ConvolutionalNetwork {
        std::fs::create_dir_all(self.write_intermediate_results.1.clone())
            .expect("Error while building the convolution product output directory ");
        ConvolutionalNetwork {
            layers: self.layers,
            outputs: self.outputs,
            write_intermediate_results: self.write_intermediate_results,
        }
    }

    pub fn network_convolve(
        &mut self,
        input: &Array2<f64>,
        optional_output_path: &str,
    ) -> Array2<f64> {
        self.outputs.push(self.layers[0].convolve(input));
        if self.layers.len() > 1 {
            for i in 1..self.layers.len() {
                let output = self.layers[i].convolve(&self.outputs[i - 1]);
                self.outputs.push(output.clone());
            }
        }
        // TO_DO: I don't believe an new intermediate product image is created for each conv layer, but currently
        // re-writes the existing one after each pass (we only end up with the final convolutional image)
        if self.write_intermediate_results.0 == true {
            for i in 0..self.outputs.len() {
                let image_name = optional_output_path.split("/").collect::<Vec<&str>>(); // .expect("Couldn't split the image path correctly");
                let isolated = image_name.last().unwrap().split(".").collect::<Vec<&str>>();
                let image_filename = isolated[isolated.len() - 2].to_owned() + &i.to_string() + ".png";
                // println!("image_filename: {}",&image_filename);
                write_result_to_file(
                    &self.outputs[i],
                    self.write_intermediate_results.1.clone() + &image_filename,
                );
            }
        }

        self.outputs.last().unwrap().to_owned()
    }
}

fn write_result_to_file(result_array: &Array2<f64>, output_path: String) {
    let (w, h) = (result_array.ncols() as u32, result_array.nrows() as u32);
    let mut img: RgbImage = ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let pixel = (result_array[[y as usize, x as usize]] * 255.) as u8;
            img.put_pixel(x, y, Rgb([pixel, pixel, pixel]));
        }
    }
    img.save(output_path.clone()).expect(
        format!(
            "Couldn't write a the intermediate result to {}",
            output_path
        )
        .as_str(),
    );
}

#[test]
#[ignore]
fn basic_conv_network() {
    let input: Array2<f64> = array![[1., 2., 3., 4.], [4., 3., 2., 1.], [1., 2., 2.5, 4.]];
    let mut conv_layers: Vec<ConvLayer> = Vec::new();
    let conv_layer_0 = ConvLayer::default(&array![[1., 0.], [0., 0.]])
        .kernel(&array![[1., 0.], [0., 0.]])
        .build();
    let conv_layer_1 = ConvLayer::default(&array![[1., 0.], [0., 0.]])
        .kernel(&array![[1., 0.], [0., 1.]])
        .build();
    conv_layers.push(conv_layer_0);
    conv_layers.push(conv_layer_1);
    let mut conv_network: ConvolutionalNetwork = ConvolutionalNetwork::default()
        .add_layers(conv_layers)
        .build();
    println!("ConvNetwork:\n{:#?}", conv_network);
    let convolved_input = conv_network.network_convolve(&input,"test_output.png");
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
