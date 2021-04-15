use mnist::*;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use tsuga::prelude::*;

fn main() {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_one_hot()
        .download_and_extract()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array4::from_shape_vec((50_000, 1, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 255.0);
    // println!("{:#.1?}\n", train_data.slice(s![image_num, .., ..]));

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 10), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);
    println!(
        "The first digit is a {:?}",
        train_labels.slice(s![image_num, ..])
    );

    let test_data = Array4::from_shape_vec((10_000, 1, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 255.);

    let test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 10), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);

    let learnrate = 0.01;
    let fc_layer_0 = FCLayer::new((784, 10), learnrate).unwrap();
    let sigmoid_layer_0 = SigmoidLayer::new();

    let mut network: Network = Network::default((1, 784), train_data, (1, 10), train_labels);
    network.add(fc_layer_0);
    network.add(sigmoid_layer_0);
    // Training iterations
    // network.info();
    network.set_iterations(1_000_000);
    network.train().expect("An error occurred while training");

    let mut total_correct = 0;
    for i in 0..10_000 {
        let input = test_data
            .slice(s![i, .., .., ..])
            .into_shape((1, 784))
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
