use ndarray::prelude::*;

extern crate tsuga;
use tsuga::prelude::*;

fn main() {
    let input: Array2<f32> = array![
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

    let mut layers_cfg: Vec<FCLayer> = Vec::new();
    let layer_0 = FCLayer::new("sigmoid", 5);
    layers_cfg.push(layer_0);

    let mut network = FullyConnectedNetwork::default(input.clone(), output.clone())
        .add_layers(layers_cfg)
        .iterations(1000)
        .build();

    let model = network.sgd_train(5);
    //println!("sgd-trained network is:\n{:#?}", network);
}
