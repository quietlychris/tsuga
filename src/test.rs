#![allow(dead_code)]
#[allow(unused_imports)]
use crate::prelude::*;
use ndarray::prelude::*;

#[test]
fn build_default_network() {
    let input: Array2<f32> = array![[1., 2., 3., 4.], [4., 3., 2., 1.], [1., 2., 2.5, 4.]];
    let output: Array2<f32> = array![[1.0, 0.0], [0., 1.0], [1.0, 0.0]];
    let mut network = FullyConnectedNetwork::default(input.clone(), output.clone())
        .iterations(10)
        .build();
    println!("Default network:\n{:#?}", network);
    let model = network.train();
}

#[test]
fn build_default_network_w_options() {
    let input: Array2<f32> = array![[1., 2., 3., 4.], [4., 3., 2., 1.], [1., 2., 2.5, 4.]];
    let output: Array2<f32> = array![[1.0, 0.0], [0., 1.0], [1.0, 0.0]];
    let mut network = FullyConnectedNetwork::default(input.clone(), output.clone())
        .iterations(10)
        .learnrate(0.0001)
        .bias_learnrate(0.001)
        .build();
    println!("Default network:\n{:#?}", network);
    let model = network.train();
}

#[test]
#[serial]
fn small_fully_connected_multi_layer() {
    let input: Array2<f32> = array![[1., 2., 3., 4.], [4., 3., 2., 1.], [1., 2., 2.5, 4.]];
    let output: Array2<f32> = array![[1.0, 0.0], [0., 1.0], [1.0, 0.0]];

    let mut layers_cfg: Vec<FCLayer> = Vec::new();
    let relu_layer_0 = FCLayer::new("relu", 5);
    layers_cfg.push(relu_layer_0);
    let sigmoid_layer_1 = FCLayer::new("sigmoid", 6);
    layers_cfg.push(sigmoid_layer_1);

    let mut network = FullyConnectedNetwork::default(input.clone(), output.clone())
        .add_layers(layers_cfg)
        .iterations(250)
        .build();

    let model = network.train();
    println!("Trained network is:\n{:#?}", network);

    let train_network_repoduced_result = model.clone().evaluate(input);

    // println!("Ideal training output:\n{:#?}",output);
    // println!("Training set fit:\n{:#?}",network.a[network.l-1]);
    assert_eq!(
        train_network_repoduced_result.mapv(|x| threshold(x, 0.5)),
        network.a[network.l - 1].mapv(|x| threshold(x, 0.5))
    );
    // println!("Reproduced trained network result from model:\n{:#?}",train_network_repoduced_result);

    let test_input: Array2<f32> = array![[4., 3., 3., 1.], [1., 2., 1., 4.]];
    let test_output: Array2<f32> = array![[0.0, 1.0], [1.0, 0.0]];
    let test_result = model.evaluate(test_input);

    // println!("Test result:\n{:#?}",test_result);
    // println!("Ideal test output:\n{:#?}",test_output);
}

use image::GenericImageView;
use std::fs;

fn build_input_and_output_matrices(data: &str) -> (Array2<f32>, Array2<f32>) {
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
    let mut output = Array::zeros((images.len(), 2));
    let mut counter = 0;
    for image in &images {
        let img = image::open(image).unwrap();
        let (w, h) = img.dimensions();
        for y in 0..h {
            for x in 0..w {
                input[[counter as usize, (y * w + x) as usize]] =
                    1.0 - (img.get_pixel(x, y)[0] as f32 / 255.);
            }
        }
        if image.contains("one") {
            output[[counter, 1]] = 1.0;
        } else {
            output[[counter, 0]] = 1.0;
        }
        counter += 1;
    }

    // Print input and output to terminal
    // println!("input:\n{:#?}",input);
    //println!("output:\n{:#?}",output);

    assert_eq!(input.shape()[0], output.shape()[0]);
    (input, output)
}

use ndarray::stack;
use rand::prelude::*;

#[test]
fn batch_sgd() {
    let mut input: Array2<f32> = array![
        [10., 11., 12., 13.],
        [20., 21., 22., 23.],
        [30., 31., 32., 33.],
        [40., 41., 42., 43.],
        [500., 510., 520., 530.],
        [600., 610., 620., 630.],
        [700., 710., 720., 730.],
        [800., 810., 820., 830.],
        [900., 910., 920., 930.],
    ];
    let output: Array2<f32> = array![
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0]
    ];

    let mut rng = thread_rng();
    let batch_size = 5;
    let mut group = Vec::new();
    for _ in 0..batch_size {
        group.push(rng.gen_range(0, input.nrows()));
    }
    println!("group: {:?}", group);

    let mut a: Array2<f32> = input
        .slice(s![group[0], ..])
        .clone()
        .to_owned()
        .into_shape((1, input.ncols()))
        .unwrap();
    println!("a_start: {:?}", a);
    for record in &group {
        let b: Array2<f32> = input
            .slice(s![*record, ..])
            .clone()
            .to_owned()
            .into_shape((1, input.ncols()))
            .unwrap();
        a = stack![Axis(0), a.clone(), b];
    }
    println!("a_end:\n{:#?}", a);
}

use ocl::Error;

#[test]
#[serial]
fn small_fully_connected_multi_layer_w_carya() -> Result<(), Error> {
    let input: Array2<f32> = array![[1., 2., 3., 4.], [4., 3., 2., 1.], [1., 2., 2.5, 4.]];
    let output: Array2<f32> = array![[1.0, 0.0], [0., 1.0], [1.0, 0.0]];

    let mut layers_cfg: Vec<FCLayer> = Vec::new();
    let sigmoid_layer_0 = FCLayer::new("sigmoid", 10);
    layers_cfg.push(sigmoid_layer_0);
    let sigmoid_layer_1 = FCLayer::new("sigmoid", 10);
    layers_cfg.push(sigmoid_layer_1);

    let mut network = FullyConnectedNetwork::default(input.clone(), output.clone())
        .add_layers(layers_cfg)
        .iterations(250)
        .bias_learnrate(0.)
        .build();

    // let model = network.train();
    let model = network.train_w_carya("GeForce")?;
    println!("Trained network is:\n{:#?}", network);

    let train_network_repoduced_result = model.clone().evaluate(input);

    // println!("Ideal training output:\n{:#?}",output);
    // println!("Training set fit:\n{:#?}",network.a[network.l-1]);
    /*
    assert_eq!(
        train_network_repoduced_result.mapv(|x| threshold(x, 0.5)),
        network.a[network.l - 1].mapv(|x| threshold(x, 0.5))
    );
    */
    // println!("Reproduced trained network result from model:\n{:#?}",train_network_repoduced_result);

    let test_input: Array2<f32> = array![[4., 3., 3., 1.], [1., 2., 1., 4.]];
    let test_output: Array2<f32> = array![[0.0, 1.0], [1.0, 0.0]];
    let test_result = model.evaluate(test_input);

    println!("Test result:\n{:#?}", test_result);
    println!("Ideal test output:\n{:#?}", test_output);
    Ok(())
}
