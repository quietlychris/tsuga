extern crate ndarray as nd;
use ndarray::prelude::*;

extern crate ndarray_stats as nds;
use crate::nds::QuantileExt;

extern crate image;
use crate::image::GenericImageView;
use std::fs;

extern crate tsuga;
use tsuga::prelude::*;

fn main() {
    let (input, output) = build_mnist_input_and_output_matrices_w_convolution("./data/012s/train");

    let mut layers_cfg: Vec<FCLayer> = Vec::new();
    let sigmoid_layer_0 = FCLayer::new("sigmoid", 4);
    layers_cfg.push(sigmoid_layer_0);
    let sigmoid_layer_1 = FCLayer::new("sigmoid", 10);
    layers_cfg.push(sigmoid_layer_1);


    let mut network = FullyConnectedNetwork::default(input.clone(), output.clone())
        .add_layers(layers_cfg)
        .iterations(1000)
        .learnrate(0.001)
        .build();

    let model = network.train();

    let (test_input, test_output) =
        build_mnist_input_and_output_matrices_w_convolution("./data/012s/test");
    let result = model.evaluate(test_input);
    //println!("test_result:\n{:#?}",result);
    let image_names = list_files("./data/012s/test");
    let mut correct_number = 0;
    for i in 0..result.shape()[0] {
        let result_row = result.slice(s![i, ..]);
        let output_row = test_output.slice(s![i, ..]);

        if (result_row.argmax() == output_row.argmax())
        //&& (result_row[result_row.argmax().unwrap()] > 0.5)
        {
            correct_number += 1;
            println!(
                "{}: {} -> result: {:.2} vs. actual {:.0}, correct #{}",
                i, image_names[i], result_row, output_row, correct_number
            );
        } else {
            println!(
                "{}: {} -> result: {:.2} vs. actual {:.0}",
                i, image_names[i], result_row, output_row
            );
        }
    }
    println!(
        "Total correct values: {}/{}, or {}%",
        correct_number,
        test_output.shape()[0],
        (correct_number as f32) / (test_output.shape()[0] as f32)
    );

}

fn build_mnist_input_and_output_matrices_w_convolution(
    directory: &str,
) -> (Array2<f64>, Array2<f64>) {
    let paths = fs::read_dir(directory)
        .expect(&format!("Couldn't index files from the {} directory", directory).to_string());
    let mut images = vec![];
    for path in paths {
        let p = path.unwrap().path().to_str().unwrap().to_string();
        images.push(p);
    }
    // println!("image list: {:?}",images); // Displays a list of the image paths, ex."./data/train/one_x.png"
    // println!("The length of images is: {}",images.len());
    let (w, h) = image::open(images[0].clone())
        .unwrap()
        .to_luma()
        .dimensions();

    let mut conv_layers: Vec<ConvLayer> = Vec::new();
    // let kernel_0 = array![[1., 0.], [0., 0.]];
    let kernel_0 = array![[1.]];
    let conv_layer_0 = ConvLayer::default(&kernel_0).build();
    conv_layers.push(conv_layer_0);
    let mut conv_network: ConvolutionalNetwork = ConvolutionalNetwork::default()
        .add_layers(conv_layers)
        .write_intermediate_results((true, "data/results/".to_string())) // If true, then supply the output directory path
        .build();

    let image_zero = image_to_array(&images[0]);
    let convolved_image_zero =
        conv_network.network_convolve(&image_zero, images[0].clone().as_str());
    let (c_n, c_m) = (convolved_image_zero.nrows(), convolved_image_zero.ncols());

    // After writing the first image to get a sense of the convoltion result, we're then turning it off
    conv_network = conv_network.write_intermediate_results((false, "".to_string()));

    let mut input = Array::zeros((images.len(), (c_n * c_m) as usize));
    let mut output = Array::zeros((images.len(), 3)); // Output is the # of records and the # of classes
    let mut counter = 0;

    for image in &images {
        let image_array = image_to_array(image);
        let convolved_image = conv_network.network_convolve(&image_array, image);
        for y in 0..convolved_image.nrows() {
            for x in 0..convolved_image.ncols() {
                input[[counter as usize, (y * c_m + x) as usize]] = convolved_image[[y, x]];
            }
        }

        if image.contains("zero") {
            output[[counter, 0]] = 1.0;
        } else if image.contains("one") {
            output[[counter, 1]] = 1.0;
        } else if image.contains("two") {
            output[[counter, 2]] = 1.0;
        } else {
            panic!("Image couldn't be classified!");
        }
        counter += 1;
    }
    assert_eq!(input.shape()[0], output.shape()[0]);
    (input, output)
}

fn image_to_array(image: &String) -> Array2<f64> {
    let img = image::open(image)
        .expect("An error occurred while open the image to convert to array for convolution");
    let (w, h) = img.dimensions();
    let mut image_array = Array::zeros((w as usize, h as usize));
    for y in 0..h {
        for x in 0..w {
            image_array[[y as usize, x as usize]] = 1. - (img.get_pixel(x, y)[0] as f64 / 255.);
        }
    }
    image_array
}

fn list_files(directory: &str) -> Vec<String> {
    let paths = fs::read_dir(directory)
        .expect(&format!("Couldn't index files from the {} directory", directory).to_string());
    let mut images = vec![];
    for path in paths {
        let p = path.unwrap().path().to_str().unwrap().to_string();
        images.push(p);
    }
    images
}
