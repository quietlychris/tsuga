extern crate ndarray as nd;
use ndarray::prelude::*;

extern crate ndarray_stats as nds;
use crate::nds::QuantileExt;

extern crate image;
use crate::image::GenericImageView;
use std::fs;

extern crate tsuga;
use tsuga::fc_layer::FCLayer;
use tsuga::fc_network::FullyConnectedNetwork;

fn main() {
    let (input, output) = build_mnist_input_and_output_matrices("./data/012s/train");

    let mut layers_cfg: Vec<FCLayer> = Vec::new();
    let sigmoid_layer_0 = FCLayer::new("sigmoid", 4);
    layers_cfg.push(sigmoid_layer_0);
    let sigmoid_layer_1 = FCLayer::new("sigmoid", 2);
    layers_cfg.push(sigmoid_layer_1);

    let mut network = FullyConnectedNetwork::default(input.clone(), output.clone())
        .add_layers(layers_cfg)
        .iterations(5000)
        .learnrate(0.001)
        .build();

    // let model = network.sgd_train(2);
    let model = network.train();

    let (test_input, test_output) = build_mnist_input_and_output_matrices("./data/012s/test");
    println!("test_output:\n{:#?}", test_output);
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
        (correct_number as f32) / (test_output.shape()[0] as f32) * 100.
    );
}

fn list_files(directory: &str) -> Vec<String> {
    let paths = fs::read_dir(directory).expect("Couldn't index files from the eight/ directory");
    let mut images = vec![];
    for path in paths {
        let p = path.unwrap().path().to_str().unwrap().to_string();
        images.push(p);
    }
    images
}

fn build_mnist_input_and_output_matrices(data: &str) -> (Array2<f64>, Array2<f64>) {
    let paths = fs::read_dir(data).expect("Couldn't index files from the eight/ directory");
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
    let mut output = Array::zeros((images.len(), 3));
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
            println!(
                "{} contains the word \"zero\" -> {}",
                image,
                output.slice(s![counter, ..])
            );
        } else if image.contains("one") {
            output[[counter, 1]] = 1.0;
            println!(
                "{} contains the word \"one\" -> {}",
                image,
                output.slice(s![counter, ..])
            )
        } else if image.contains("two") {
            output[[counter, 2]] = 1.0;
            println!(
                "{} contains the word \"two\" -> {}",
                image,
                output.slice(s![counter, ..])
            );
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
