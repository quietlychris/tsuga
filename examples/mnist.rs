extern crate ndarray as nd;
use ndarray::prelude::*;

extern crate ndarray_stats as nds;
use crate::nds::QuantileExt;

extern crate image;
use crate::image::{DynamicImage, GenericImageView, ImageBuffer};

extern crate imageproc;
use imageproc::edges::canny;

use std::fs;

extern crate tsuga;
use tsuga::prelude::*;

fn main() {
    let (input, output) = build_mnist_input_and_output_matrices("./data/mnist/train");

    let mut layers_cfg: Vec<FCLayer> = Vec::new();
    let sigmoid_layer_0 = FCLayer::new("sigmoid", 800);
    layers_cfg.push(sigmoid_layer_0);

    let mut network = FullyConnectedNetwork::default(input, output)
        .add_layers(layers_cfg)
        .iterations(10)
        .learnrate(0.0001)
        .bias_learnrate(0.1)
        .error_threshold(50.)
        .min_iterations(100)
        .build();

    let model = network.train();
    // let model = network.train_w_carya("GeForce").unwrap();
    // GPU last trained on learnrate = 0.000025, iterations = 1000 for ~72%
    // let model = network.train_on_gpu("GeForce");
    // let model = network.sgd_train(200);

    let (test_input, test_output) = build_mnist_input_and_output_matrices("./data/mnist/test");

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
            /*println!(
                    "{}: {} -> result: {:.2} vs. actual {:.0}, correct #{}",
                    i, image_names[i], result_row, output_row, correct_number
                );
            } else {
                println!(
                    "{}: {} -> result: {:.2} vs. actual {:.0}",
                    i, image_names[i], result_row, output_row
                );*/
        }
    }
    println!(
        "Total correct values: {}/{}, or {}%",
        correct_number,
        test_output.shape()[0],
        (correct_number as f32) * 100. / (test_output.shape()[0] as f32)
    );
}

fn build_mnist_input_and_output_matrices(directory: &str) -> (Array2<f32>, Array2<f32>) {
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
    let mut output = Array::zeros((images.len(), 10)); // Output is the # of records and the # of classes
    let mut counter = 0;

    for image in &images {
        let image_array = image_to_array(image);
        for y in 0..image_array.nrows() {
            for x in 0..image_array.ncols() {
                input[[counter as usize, (y * (w as usize) + x) as usize]] = image_array[[y, x]];
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
    let mut img: DynamicImage = image::open(image)
        .expect("An error occurred while open the image to convert to array for convolution");

    //let canny_image = canny(&img.to_luma(), 100., 200.);
    //img = image::DynamicImage::ImageLuma8(canny_image);

    //println!("About to save canny image!");
    //img.save("./data/results/canny.png").expect("Couldn't save the canny image");

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
