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
    let (input, output) = build_mnist_input_and_output_matrices_w_convolution("./data/mnist/train");

    let mut layers_cfg: Vec<FCLayer> = Vec::new();

    let mut network = FullyConnectedNetwork::default(input, output)
        .add_layers(layers_cfg)
        .iterations(200)
        //.learnrate(0.018)
        .learnrate(0.0008)
        .bias_learnrate(0.)
        .build();

    // println!("Networks layers_cfg:\n{:#?}", network.layers_cfg);

    network.print_shape();

    let model = network.train();
    //let model = network.train_on_gpu("GeForce");

    let (test_input, test_output) =
        build_mnist_input_and_output_matrices_w_convolution("./data/mnist/test");

    println!("About to evaluate the conv_mnist model:");
    let result = model.evaluate(test_input);

    // println!("test_result:\n{:#?}", result);
    let image_names = list_files("./data/mnist/test");
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
        (correct_number as f32) * 100. / (test_output.shape()[0] as f32)
    );
}

fn build_mnist_input_and_output_matrices_w_convolution(
    directory: &str,
) -> (Array2<f32>, Array2<f32>) {
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

    let kernel_0 = array![[-1., 1., -1.], [1., 2., -1.], [-1., 1., -1.]]; // Strong
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
    let mut output = Array::zeros((images.len(), 10)); // Output is the # of records and the # of classes
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
        } else if image.contains("three") {
            output[[counter, 3]] = 1.0;
        } else if image.contains("four") {
            output[[counter, 4]] = 1.0;
        } else if image.contains("five") {
            output[[counter, 5]] = 1.0;
        } else if image.contains("six") {
            output[[counter, 6]] = 1.0;
        } else if image.contains("seven") {
            output[[counter, 7]] = 1.0;
        } else if image.contains("eight") {
            output[[counter, 8]] = 1.0;
        } else if image.contains("nine") {
            output[[counter, 9]] = 1.0;
        } else {
            panic!(format!("Image {} couldn't be classified!", image));
        }
        counter += 1;
    }
    assert_eq!(input.shape()[0], output.shape()[0]);
    (input, output)
}

fn image_to_array(image: &String) -> Array2<f32> {
    let img = image::open(image)
        .expect("An error occurred while open the image to convert to array for convolution");
    let (w, h) = img.dimensions();
    let mut image_array = Array::zeros((w as usize, h as usize));
    for y in 0..h {
        for x in 0..w {
            if img.get_pixel(x, y)[0] > 1 {
                image_array[[y as usize, x as usize]] = 1.0;
            } else {
                image_array[[y as usize, x as usize]] = 0.;
            }
            //image_array[[y as usize, x as usize]] = 1.0 - (img.get_pixel(x, y)[0] as f32 / 255.);
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
