use crate::fc_layer::*;

pub fn activation_function(layers_cfg: &Vec<FCLayer>, i: usize, x: f64) -> f64 {
    /*let alert = match &layers_cfg[i].activation_function {
        sigmoid_function => println!("Applying a sigmoid function"),
        relu_function => println!("Applying a relu function"),
        _ => panic!("The specified activation function does not exist!")
    };*/
    //let layer_activation_fn = layers_cfg[i].activation_function;
    let var = match &*layers_cfg[i].activation_function {
        "sigmoid" => sigmoid(x),
        "relu" => relu(x),
        "linear" => x,
        _ => panic!("The specified activation function does not exist!"),
    };
    var
}

pub fn activation_function_prime(layers_cfg: &Vec<FCLayer>, i: usize, x: f64) -> f64 {
    // let layer_activation_fn = layers_cfg[i].activation_function;

    let var = match &*layers_cfg[i].activation_function {
        "sigmoid" => sigmoid_prime(x),
        "relu" => relu_prime(x),
        "linear" => 1.,
        _ => panic!("The specified activation function does not exist!"),
    };
    var
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_prime(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub fn relu(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

pub fn relu_prime(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        1.
    }
}

pub fn threshold(x: f64, threshold: f64) -> f64 {
    if x > threshold {
        1.0
    } else {
        0.0
    }
}
