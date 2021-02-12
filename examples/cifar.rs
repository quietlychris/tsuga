use cifar_ten::*;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use tsuga::prelude::*;

<<<<<<< HEAD
fn main() {
    let (train_data, train_labels, test_data, test_labels) = Cifar10::default()
        .show_images(false) // won't display a window with the image
        .download_and_extract(true) // must enable the "download" feature
        .normalize(true) // floating point values will be normalized across [0, 1.0]
        .build_f32()
        .expect("Failed to build CIFAR-10 data");

    let learnrate = 0.015;
    let fc_layer_0 = FCLayer::new((3072, 300), "sigmoid".to_string(), learnrate).unwrap();
    let fc_layer_1 = FCLayer::new((300, 10), "sigmoid".to_string(), learnrate).unwrap();

    let mut network: Network = Network::default((1, 3072), train_data, (1, 10), train_labels);
    network.add(fc_layer_0);
    network.add(fc_layer_1);
    // Training iterations
    // network.info();
    network.set_iterations(10_000);
    network.train().expect("An error occurred while training");

    let mut total_correct = 0;
    for i in 0..10_000 {
        let input = test_data
            .slice(s![i, .., .., ..])
            .into_shape((1, 3072))
            .unwrap()
            .to_owned();
        let actual = test_labels.slice(s![i, ..]).into_shape((1, 10)).unwrap();

        let mut output = network.evaluate(input).unwrap();
        softmax(&mut output);

        if actual.argmax() == output.argmax() {
            total_correct += 1;
=======
// Expects the unpacked CIFAR-10 binary data to be located in the
// ./data/cifar-10-batches-bin directory
// The dataset can be downloaded here: https://www.cs.toronto.edu/~kriz/cifar.html

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let labels = &[
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

    let (train_data, train_labels, test_data, test_labels) = Cifar10::default()
        .show_images(false) // won't display a window with the image
        .download_and_extract(true) // must enable the "download" feature
        .normalize(true) // floating point values will be normalized across [0, 1.0]
        .build_f32()
        .expect("Failed to build CIFAR-10 data");

    let train_data = train_data.into_shape((50_000, 3 * 32 * 32))?;
    let test_data = test_data.into_shape((10_000, 3 * 32 * 32))?;

    let mut layers_cfg: Vec<FCLayer> = Vec::new();
    let relu_layer_0 = FCLayer::new("sigmoid", 600);
    layers_cfg.push(relu_layer_0);
    let sigmoid_layer_1 = FCLayer::new("sigmoid", 256);
    layers_cfg.push(sigmoid_layer_1);

    let mut network = FullyConnectedNetwork::default(train_data, train_labels)
        .add_layers(layers_cfg)
        .validation_pct(0.01)
        .error_threshold(1e-9)
        .iterations(5_000)
        .min_iterations(1_000)
        .learnrate(0.001)
        .build();

    network.train()?;

    println!("About to evaluate the CIFAR-10 model:");
    let test_result = network.evaluate(test_data.clone());
    compare_results(test_result.clone(), test_labels.clone());

    let mut rng = rand::thread_rng();
    let num: usize = rng.gen_range(0..test_data.nrows());
    println!(
        "Test result #{} (proper label: {}) has a classification spread of:\n------------------------------",
        num, labels[test_labels.slice(s![num, ..]).argmax()?]
    );
    for i in 0..labels.len() {
        println!("{}: {:.2}%", labels[i], test_result[[num, i]] * 100.);
    }
    let test_result_img = rgb_ndarray3_to_rgb_image(
        test_data
            .slice(s![num, ..])
            .to_owned()
            .into_shape((3, 32, 32))
            .unwrap(),
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

fn compare_results(actual: Array2<f32>, ideal: Array2<f32>) {
    let mut correct_number = 0;
    for i in 0..actual.nrows() {
        let result_row = actual.slice(s![i, ..]);
        let output_row = ideal.slice(s![i, ..]);

        if result_row.argmax() == output_row.argmax() {
            correct_number += 1;
>>>>>>> 89a747c01... Update dependencies, refactor CIFAR-10 example
        }
    }
    println!(
        "Total correct: {}/10_000  ({:.2}%)",
        total_correct,
        (total_correct as f32 / 10_000.) * 100.
    );
}
