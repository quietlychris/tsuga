use mnist::*;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
<<<<<<< HEAD
use tsuga::prelude::*;
=======
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
    let num: usize = rng.gen_range(0..test_result.nrows());
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


    println!("\nPlease hit [ ESC ] to quit window:");
    let window_options = WindowOptions {
        name: "image".to_string(),
        size: [100, 100],
        resizable: true,
        preserve_aspect_ratio: true,
    };
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
>>>>>>> 89a747c01... Update dependencies, refactor CIFAR-10 example

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
    let fc_layer_0 = FCLayer::new((784, 300), "sigmoid".to_string(), learnrate).unwrap();
    let fc_layer_1 = FCLayer::new((300, 64), "sigmoid".to_string(), learnrate).unwrap();
    let fc_layer_2 = FCLayer::new((64, 10), "sigmoid".to_string(), learnrate).unwrap();

    let mut network: Network = Network::default((1, 784), train_data, (1, 10), train_labels);
    network.add(fc_layer_0);
    network.add(fc_layer_1);
    network.add(fc_layer_2);
    // Training iterations
    // network.info();
    network.set_iterations(10_000);
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
