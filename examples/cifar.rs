use cifar_ten::*;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use tsuga::prelude::*;

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
        }
    }
    println!(
        "Total correct: {}/10_000  ({:.2}%)",
        total_correct,
        (total_correct as f32 / 10_000.) * 100.
    );
}
