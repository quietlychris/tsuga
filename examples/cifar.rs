use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

use cifar_10::*;
use minifb::{Key, ScaleMode, Window, WindowOptions};
use rand::prelude::*;
use tsuga::prelude::*;

// Expects the unpacked CIFAR-10 binary data to be located in the
// ./data/cifar-10-batches-bin directory
// The dataset can be downloaded here: https://www.cs.toronto.edu/~kriz/cifar.html

const LABELS: &[&'static str] = &[
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (mut train_data, train_labels, mut test_data, test_labels) = Cifar10::default()
        .show_images(false)
        .build_as_flat_f32()
        .expect("Failed to build CIFAR-10 data");

    train_data.mapv(|x| x / 256.);
    test_data.mapv(|x| x / 256.);

    let mut rng = rand::thread_rng();
    let mut num: usize = rng.gen_range(0, test_data.nrows());
    println!(
        "Train record #{} has a label of {}",
        num,
        return_label_from_one_hot(train_labels.slice(s![num, ..]))
    );
    display_img(
        &train_data
            .clone()
            .into_shape((train_data.nrows(), 3, 32, 32))?,
        &train_labels.to_owned(),
        num,
    );

    let mut layers_cfg: Vec<FCLayer> = Vec::new();
    let relu_layer_0 = FCLayer::new("sigmoid", 1000);
    layers_cfg.push(relu_layer_0);
    let sigmoid_layer_1 = FCLayer::new("sigmoid", 350);
    layers_cfg.push(sigmoid_layer_1);
    let sigmoid_layer_2 = FCLayer::new("sigmoid", 100);
    layers_cfg.push(sigmoid_layer_2);

    let mut network = FullyConnectedNetwork::default(train_data, train_labels)
        .add_layers(layers_cfg)
        .iterations(5_000)
        .learnrate(0.005)
        .build();

    network.train()?;

    println!("About to evaluate the CIFAR-10 model:");
    let test_result = network.evaluate(test_data.clone());
    compare_results(test_result.clone(), test_labels.clone());

    num = rng.gen_range(0, test_data.nrows());
    println!(
        "Test result #{} has a classification spread of:\n------------------------------",
        num
    );
    for i in 0..LABELS.len() {
        println!("{}: {:.2}%", LABELS[i], test_result[[num, i]] * 100.);
    }
    display_img(
        &test_data
            .clone()
            .into_shape((test_data.nrows(), 3, 32, 32))?,
        &test_result.to_owned(),
        num,
    );

    Ok(())
}

fn compare_results(actual: Array2<f32>, ideal: Array2<f32>) {
    let mut correct_number = 0;
    for i in 0..actual.nrows() {
        let result_row = actual.slice(s![i, ..]);
        let output_row = ideal.slice(s![i, ..]);

        if result_row.argmax() == output_row.argmax() {
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

fn display_img(data: &Array4<f32>, labels: &Array2<f32>, num: usize) {
    // Displaying in minifb window instead of saving as a .png
    let img_arr = data.slice(s!(num, .., .., ..));
    let mut img_vec: Vec<u32> = Vec::with_capacity(32 * 32);
    let (w, h) = (32, 32);
    for y in 0..h {
        for x in 0..w {
            let temp: [u8; 4] = [
                (img_arr[[2, y, x]] * 255.) as u8,
                (img_arr[[1, y, x]] * 255.) as u8,
                (img_arr[[0, y, x]] * 255.) as u8,
                255u8,
            ];
            // println!("temp: {:?}", temp);
            img_vec.push(u32::from_le_bytes(temp));
        }
    }
    println!(
        "Data label: {}",
        return_label_from_one_hot(labels.slice(s![num, ..]))
    );
    display_in_window(img_vec);
}

fn display_in_window(buffer: Vec<u32>) {
    let (window_width, window_height) = (600, 600);
    let mut window = Window::new(
        "Test - ESC to exit",
        window_width,
        window_height,
        WindowOptions {
            resize: true,
            scale_mode: ScaleMode::Center,
            ..WindowOptions::default()
        },
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // Limit to max ~60 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    while window.is_open() && !window.is_key_down(Key::Escape) && !window.is_key_down(Key::Q) {
        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window.update_with_buffer(&buffer, 32, 32).unwrap();
    }
}

fn return_label_from_one_hot(one_hot: ArrayView1<f32>) -> String {
    let max_index = one_hot.argmax().unwrap();
    let mut one_hot = one_hot.mapv(|x| x as u8);
    one_hot[max_index] = 1;
    if one_hot == array![1, 0, 0, 0, 0, 0, 0, 0, 0, 0] {
        "airplane".to_string()
    } else if one_hot == array![0, 1, 0, 0, 0, 0, 0, 0, 0, 0] {
        "automobile".to_string()
    } else if one_hot == array![0, 0, 1, 0, 0, 0, 0, 0, 0, 0] {
        "bird".to_string()
    } else if one_hot == array![0, 0, 0, 1, 0, 0, 0, 0, 0, 0] {
        "cat".to_string()
    } else if one_hot == array![0, 0, 0, 0, 1, 0, 0, 0, 0, 0] {
        "deer".to_string()
    } else if one_hot == array![0, 0, 0, 0, 0, 1, 0, 0, 0, 0] {
        "dog".to_string()
    } else if one_hot == array![0, 0, 0, 0, 0, 0, 1, 0, 0, 0] {
        "frog".to_string()
    } else if one_hot == array![0, 0, 0, 0, 0, 0, 0, 1, 0, 0] {
        "horse".to_string()
    } else if one_hot == array![0, 0, 0, 0, 0, 0, 0, 0, 1, 0] {
        "ship".to_string()
    } else if one_hot == array![0, 0, 0, 0, 0, 0, 0, 0, 0, 1] {
        "truck".to_string()
    } else {
        format!("Error: no valid label could be assigned to {}", one_hot).to_string()
    }
}
