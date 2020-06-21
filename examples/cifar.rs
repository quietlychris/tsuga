extern crate ndarray as nd;
use ndarray::prelude::*;

extern crate ndarray_stats as nds;
use crate::nds::QuantileExt;

extern crate image;
use crate::image::{DynamicImage, GenericImageView, ImageBuffer};

extern crate imageproc;
use imageproc::edges::canny;
use imageproc::map::*;

use std::fs;

extern crate tsuga;
use tsuga::prelude::*;

fn main() {
    let (input, output) = build_cifar_input_and_output_matrices("./data/cifar/train");

    // let mut layers_cfg: Vec<FCLayer> = Vec::new();
    // let sigmoid_layer_0 = FCLayer::new("sigmoid",500);
    // layers_cfg.push(sigmoid_layer_0);

    let mut network = FullyConnectedNetwork::default(input, output)
        // .add_layers(layers_cfg)
        .iterations(500)
        .learnrate(0.0005)
        .bias_learnrate(0.00)
        .build();

    let model = network.train();
    // GPU last trained on learnrate = 0.000025, iterations = 1000 for ~72%
    // let model = network.train_on_gpu("GeForce");

    let (test_input, test_output) = build_cifar_input_and_output_matrices("./data/cifar/test");

    println!("About to evaluate the conv_mnist model:");
    let result = model.evaluate(test_input);

    // println!("test_result:\n{:#?}", result);
    let image_names = list_files("./data/cifar/test");
    let mut correct_number = 0;
    for i in 0..result.shape()[0] {
        let result_row = result.slice(s![i, ..]);
        let output_row = test_output.slice(s![i, ..]);

        if (result_row.argmax() == output_row.argmax()) {
            correct_number += 1;
        /*println!(
            "{}: {} -> result: {:.2} vs. actual {:.0}, correct #{}",
            i, image_names[i], result_row, output_row, correct_number
        );*/
        } else {
            /*println!(
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

fn build_cifar_input_and_output_matrices(directory: &str) -> (Array2<f32>, Array2<f32>) {
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

    let mut input = Array::zeros((images.len(), (w * h) as usize * 3)); // Times 3 because of RGB channels
    let mut output = Array::zeros((images.len(), 10)); // Output is the # of records and the # of classes
    let mut counter = 0;

    for image in &images {
        let image_array = image_to_array(image);
        let (rows, cols, depth) = (
            image_array.dim().0,
            image_array.dim().1,
            image_array.dim().2,
        );
        // println!("image array has dimensions of ({},{},{})",rows,cols,depth);
        for y in 0..rows {
            for x in 0..cols {
                input[[counter as usize, (y * (w as usize) + x) as usize]] = image_array[[y, x, 0]];
                input[[counter as usize, (y * (w as usize) + x) as usize * 2]] =
                    image_array[[y, x, 1]];
                input[[counter as usize, (y * (w as usize) + x) as usize * 3]] =
                    image_array[[y, x, 2]];
            }
        }

        if image.contains("airplane") {
            output[[counter, 0]] = 1.0;
        } else if image.contains("automobile") {
            output[[counter, 1]] = 1.0;
        } else if image.contains("bird") {
            output[[counter, 2]] = 1.0;
        } else if image.contains("cat") {
            output[[counter, 3]] = 1.0;
        } else if image.contains("deer") {
            output[[counter, 4]] = 1.0;
        } else if image.contains("dog") {
            output[[counter, 5]] = 1.0;
        } else if image.contains("frog") {
            output[[counter, 6]] = 1.0;
        } else if image.contains("horse") {
            output[[counter, 7]] = 1.0;
        } else if image.contains("ship") {
            output[[counter, 8]] = 1.0;
        } else if image.contains("truck") {
            output[[counter, 9]] = 1.0;
        } else {
            panic!(format!("Image {} couldn't be classified!", image));
        }
        counter += 1;
    }
    assert_eq!(input.shape()[0], output.shape()[0]);
    (input, output)
}

fn image_to_array(image: &String) -> Array3<f32> {
    let mut img = image::open(image)
        .expect("An error occurred while open the image to convert to array for convolution");

    let mut red_img = red_channel(&img.to_rgb());
    let mut green_img = green_channel(&img.to_rgb());
    let mut blue_img = blue_channel(&img.to_rgb());

    //let red_canny = canny(&red_img, 100., 200.);
    //let green_canny = canny(&green_img, 100., 200.);
    //let blue_canny = canny(&blue_img, 100., 200.);

    //let red_img = image::DynamicImage::ImageLuma8(red_canny);
    //let green_img = image::DynamicImage::ImageLuma8(green_canny);
    //let blue_img = image::DynamicImage::ImageLuma8(blue_canny);

    // println!("About to save canny image!");
    //red_img.save("./data/results/red_canny.png").expect("Couldn't save the canny image");
    //green_img.save("./data/results/green_canny.png").expect("Couldn't save the canny image");
    //blue_img.save("./data/results/blue_canny.png").expect("Couldn't save the canny image");

    let (w, h) = img.dimensions();

    let mut image_array = Array::zeros((w as usize, h as usize, 3));
    for y in 0..h {
        for x in 0..w {
            if red_img.get_pixel(x, y)[0] > 1 {
                image_array[[y as usize, x as usize, 0]] = 1.0;
            } else {
                image_array[[y as usize, x as usize, 0]] = 0.;
            }

            if green_img.get_pixel(x, y)[0] > 1 {
                image_array[[y as usize, x as usize, 1]] = 1.0;
            } else {
                image_array[[y as usize, x as usize, 1]] = 0.;
            }

            if blue_img.get_pixel(x, y)[0] > 1 {
                image_array[[y as usize, x as usize, 2]] = 1.0;
            } else {
                image_array[[y as usize, x as usize, 2]] = 0.;
            }
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
