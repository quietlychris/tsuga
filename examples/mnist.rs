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
    let (input, output) =
        build_mnist_input_and_output_matrices_w_convolution("./data/full_mnist/train_subset");

    let mut layers_cfg: Vec<FCLayer> = Vec::new();
    let sigmoid_layer_0 = FCLayer::new("sigmoid", 2);
    layers_cfg.push(sigmoid_layer_0);
    let sigmoid_layer_1 = FCLayer::new("sigmoid", 2);
    layers_cfg.push(sigmoid_layer_1);
    let sigmoid_layer_2 = FCLayer::new("sigmoid", 2);
    layers_cfg.push(sigmoid_layer_2);

    let mut network = FullyConnectedNetwork::default(input.clone(), output.clone())
        .add_layers(layers_cfg)
        .iterations(250)
        .learnrate(0.0003)
        .build();

    let model = network.train();

    let (test_input, test_output) =
        build_mnist_input_and_output_matrices_w_convolution("./data/full_mnist/test_subset");
    let result = model.evaluate(test_input);
    //println!("test_result:\n{:#?}",result);
    let image_names = list_files("./data/full_mnist/test_subset");
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

    let image_zero = image_to_array(&images[0]);
    let kernel_0 = array![[-1., 0., 0.], [0., 0., 0.], [0., 0., 0.]];
    let kernel_1 = array![[1.,0.],[0.,0.]];
    let mut conv_layers: Vec<ConvLayer> = Vec::new();
    let conv_layer_0 = ConvLayer::default(&kernel_0).build();
    let conv_layer_1 = ConvLayer::default(&kernel_1).build();
    conv_layers.push(conv_layer_0);
    conv_layers.push(conv_layer_1);
    let mut conv_network: ConvolutionalNetwork = ConvolutionalNetwork::default(image_zero)
        .add_layers(conv_layers)
        .build();
    let convolved_image_zero = conv_network.network_convolve();
    let (c_n, c_m) = (convolved_image_zero.nrows(), convolved_image_zero.ncols());

    let mut input = Array::zeros((images.len(), (c_n * c_m) as usize));
    let mut output = Array::zeros((images.len(), 4));
    let mut counter = 0;

    for image in &images {
        let image_array = image_to_array(image);

        let convolved_image = conv_network.network_convolve();
        for y in 0..convolved_image.nrows() {
            for x in 0..convolved_image.ncols() {
                input[[counter as usize, (y * c_m + x) as usize]] =
                    convolved_image[[y,x]];
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
        } else {
            panic!("Image couldn't be classified!");
        }
        counter += 1;
    }
    assert_eq!(input.shape()[0], output.shape()[0]);
    (input, output)
}

fn image_to_array(image: &String) -> Array2<f64> {
    let img = image::open(image).unwrap();
    let (w, h) = img.dimensions();
    let mut image_array = Array::zeros((28, 28));
    for y in 0..h {
        for x in 0..w {
            image_array[[y as usize, x as usize]] = 1.0 - (img.get_pixel(x, y)[0] as f64 / 255.);
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

fn build_mnist_input_and_output_matrices(directory: &str) -> (Array2<f64>, Array2<f64>) {
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

    let mut input = Array::zeros((images.len(), (w * h) as usize));
    let mut output = Array::zeros((images.len(), 4));
    let mut counter = 0;

    for image in &images {
        let img = image::open(image).unwrap();
        let (w, h) = img.dimensions();
        for y in 0..h {
            for x in 0..w {
                input[[counter as usize, (y * w + x) as usize]] =
                    1.0 - (img.get_pixel(x, y)[0] as f64 / 255.);
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
        } else {
            panic!("Image couldn't be classified!");
        }
        // println!("{} -> {}", image, output.slice(s![counter, ..]));
        counter += 1;
    }

    // Print input and output to terminal
    // println!("input:\n{:#?}",input);
    //println!("output:\n{:#?}",output);

    assert_eq!(input.shape()[0], output.shape()[0]);
    (input, output)
}
