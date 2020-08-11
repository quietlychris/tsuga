extern crate ndarray as nd;
use ndarray::prelude::*;
extern crate ndarray_stats as nds;
use crate::nds::QuantileExt;
extern crate image;

extern crate tsuga;
use tsuga::prelude::*;
extern crate cifar_10;
use cifar_10::*;

// Expects the unpacked CIFAR-10 binary data to be located in the 
// ./data/cifar-10-batches-bin directory
// The dataset can be downloaded here: https://www.cs.toronto.edu/~kriz/cifar.html

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let (mut train_data, train_labels, mut test_data, test_labels) = Cifar10::default()
        .show_images(false)
        .build_as_flat_f32()
        .expect("Failed to build CIFAR-10 data");
    
    train_data.mapv(|x| x / 256.);
    test_data.mapv(|x| x / 256.);

    let mut layers_cfg: Vec<FCLayer> = Vec::new();
    let relu_layer_0 = FCLayer::new("relu", 600);
    layers_cfg.push(relu_layer_0);
    let sigmoid_layer_1 = FCLayer::new("sigmoid", 350);
    layers_cfg.push(sigmoid_layer_1);
    let sigmoid_layer_2 = FCLayer::new("sigmoid", 100);
    layers_cfg.push(sigmoid_layer_2);

    let mut network = FullyConnectedNetwork::default(train_data, train_labels)
        .add_layers(layers_cfg)
        .iterations(1000)
        .learnrate(0.0005)
        .build();

    network.train();

    println!("About to evaluate the CIFAR-10 model:");
    let test_result = network.evaluate(test_data);
    compare_results(test_result, test_labels);
    Ok(())
}

fn compare_results(mut actual: Array2<f32>, ideal: Array2<f32>) {
    softmax(&mut actual);
    let mut correct_number = 0;
    for i in 0..actual.nrows() {
        let result_row = actual.slice(s![i, ..]);
        let output_row = ideal.slice(s![i, ..]);

        if (result_row.argmax() == output_row.argmax()) {
            correct_number += 1;
        }
    }
    println!(
        "Total correct values: {}/{}, or {}%",
        correct_number,
        actual.nrows(),
        (correct_number as f32) * 100. / (actual.nrows() as f32)
    );
}
