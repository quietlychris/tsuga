extern crate ndarray as nd;
use ndarray::prelude::*;

extern crate tsuga;
use tsuga::prelude::*;

fn main() {
    let input: Array2<f32> = array![[1., 2., 3., 4.], [4., 3., 2., 1.], [1., 2., 2.5, 4.]];
    let output: Array2<f32> = array![[1.0, 0.0], [0., 1.0], [1.0, 0.0]];

    //let mut layers_cfg: Vec<FCLayer> = Vec::new();
    //let relu_layer_0 = FCLayer::new("relu", 5);
    //layers_cfg.push(relu_layer_0);
    //let sigmoid_layer_1 = FCLayer::new("sigmoid", 6);
    //layers_cfg.push(sigmoid_layer_1);

    let mut network = FullyConnectedNetwork::default(input.clone(), output.clone())
        //.add_layers(layers_cfg)
        .iterations(100)
        .build();

    let model = network.train_on_gpu("Intel");
    //println!("Trained network is:\n{:#?}", network);

    /*let train_network_repoduced_result = model.clone().evaluate(input);

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
    */
}
