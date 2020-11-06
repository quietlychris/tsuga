use ndarray::prelude::*;
use tsuga::prelude::*;

use mnist::*;
use ndarray_stats::QuantileExt;
use rand::prelude::*;
use show_image::{make_window_full, Event, WindowOptions};

const MNIST_TYPE: &str = "fashion"; // pick "standard" or "fashion"

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let labels = match MNIST_TYPE {
        "standard" => &["0 ", "1 ", "2 ", "3 ", "4 ", "5 ", "6 ", "7 ", "8 ", "9 "],
        "fashion" => &[
            "T-shirt",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ],
        _ => panic!("Please make sure the specified type is either 'fashion' or 'standard"),
    };
    let (input, output, test_input, test_output) = mnist_as_ndarray();
    println!("Successfully unpacked the MNIST dataset into Array2<f32> format!");

    // Now we can begin configuring any additional hidden layers, specifying their size and activation function
    let mut layers_cfg: Vec<FCLayer> = Vec::new();
    let sigmoid_layer_0 = FCLayer::new("sigmoid", 128);
    layers_cfg.push(sigmoid_layer_0);
    let sigmoid_layer_1 = FCLayer::new("sigmoid", 64);
    layers_cfg.push(sigmoid_layer_1);

    // The network can now be built using the specified layer configurations
    // Several other options for tuning the network's performance are available as well
    let mut fcn = FullyConnectedNetwork::default(input, output)
        .add_layers(layers_cfg)
        .iterations(10_000)
        .min_iterations(700)
        .error_threshold(0.05)
        .learnrate(0.01)
        .batch_size(200)
        .validation_pct(0.0001)
        .build();

    // Training occurs in place on the network
    fcn.train().expect("An error occurred while training");

    // We can now pass an appropriately-sized input through our trained network,
    // receiving an Array2<f32> on the output
    let test_result = fcn.evaluate(test_input.clone());

    // And will compare that output against the ideal one-hot encoded testing label array
    compare_results(test_result.clone(), test_output);

    // Now display a singular value with the classification spread to see an example of the actual values
    let mut rng = rand::thread_rng();
    let num: usize = rng.gen_range(0, test_result.nrows());
    println!(
        "Test result #{} has a classification spread of:\n------------------------------",
        num
    );
    for i in 0..labels.len() {
        println!("{}: {:.2}%", labels[i], test_result[[num, i]] * 100.);
    }

    let test_result_img = bw_ndarray2_to_rgb_image(
        test_input
            .slice(s![num, ..])
            .to_owned()
            .into_shape((28, 28))
            .expect("Couldn't put into 28x28"),
    );

    let window_options = WindowOptions {
        name: "image".to_string(),
        size: [100, 100],
        resizable: true,
        preserve_aspect_ratio: true,
    };
    println!("\nPlease hit [ ESC ] to quit window:");
    let window = make_window_full(window_options).unwrap();
    window.set_image(test_result_img, "test_result").unwrap();

    for event in window.events() {
        if let Event::KeyboardEvent(event) = event {
            if event.key == show_image::KeyCode::Escape {
                break;
            }
        }
    }

    show_image::stop()?;
    Ok(())
}

fn mnist_as_ndarray() -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let (trn_size, _rows, _cols) = (60_000, 28, 28);
    let tst_size = 10_000;

    // Deconstruct the returned Mnist struct.
    // You can see the default Mnist struct at https://docs.rs/mnist/0.4.0/mnist/struct.MnistBuilder.html
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = match MNIST_TYPE {
        "standard" => MnistBuilder::new()
            .base_path("data/mnist")
            .label_format_one_hot()
            .download_and_extract()
            .finalize(),
        "fashion" => MnistBuilder::new()
            .base_path("data/fashion")
            .label_format_one_hot()
            .download_and_extract()
            .finalize(),
        _ => panic!("Please make sure the specified type is either 'fashion' or 'standard"),
    };

    // Convert the returned Mnist struct to Array2 format
    let trn_lbl: Array2<f32> = Array2::from_shape_vec((trn_size, 10), trn_lbl)
        .expect("Error converting labels to Array2 struct")
        .map(|x| *x as f32);
    // println!("The first digit is a {:?}",trn_lbl.slice(s![image_num, ..]) );

    // Can use an Array2 or Array3 here (Array3 for visualization)
    let trn_img = Array2::from_shape_vec((trn_size, 784), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);
    // println!("{:#.0}\n",trn_img.slice(s![image_num, .., ..]));

    // Convert the returned Mnist struct to Array2 format
    let tst_lbl: Array2<f32> = Array2::from_shape_vec((tst_size, 10), tst_lbl)
        .expect("Error converting labels to Array2 struct")
        .map(|x| *x as f32);

    let tst_img = Array2::from_shape_vec((tst_size, 784), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    (trn_img, trn_lbl, tst_img, tst_lbl)
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
