use crate::activation_functions::*;
use crate::fc_layer::*;
use crate::fc_model::*;
use crate::*;

use image::*;
use rand::prelude::*;

// TO_DO: The NN fields are all currently public, but this might not be required as a final configuration
#[derive(Debug, Clone)]
pub struct FullyConnectedNetwork {
    pub layers_cfg: Vec<FCLayer>,
    pub z: Vec<Array2<f64>>,     // intermediate matrix products
    pub w: Vec<Array2<f64>>,     // weight matrices
    pub a: Vec<Array2<f64>>,     // output layers
    pub delta: Vec<Array2<f64>>, // the delta matrix for backpropogation
    pub output: Array2<f64>,     // The target output layer
    pub l: usize,                // number of layers in the neural network
    pub learnrate: f64,          // learnrate of the network, often "alpha" in equations
    pub iterations: usize,       // number of training iterations
}

impl FullyConnectedNetwork {
    pub fn update_weights(&mut self, w: Vec<Array2<f64>>) {
        self.w = w;
    }

    pub fn learnrate(mut self, learnrate: f64) -> Self {
        self.learnrate = learnrate;
        self
    }

    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn build(self) -> FullyConnectedNetwork {
        FullyConnectedNetwork {
            layers_cfg: self.layers_cfg,
            z: self.z,
            w: self.w,
            a: self.a,
            delta: self.delta,
            output: self.output,
            l: self.l,
            learnrate: self.learnrate,
            iterations: self.iterations,
        }
    }

    pub fn default(input: Array2<f64>, output: Array2<f64>) -> Self {
        let (o_n, o_m) = (output.nrows(), output.ncols());
        let network = FullyConnectedNetwork {
            layers_cfg: vec![FCLayer::new("sigmoid", o_n)],
            z: vec![Array::zeros((o_n, o_m))],
            w: vec![Array::random(
                (input.ncols(), output.ncols()),
                Uniform::new(-0.5, 0.5),
            )],
            a: vec![input.clone(), Array::zeros((o_n, o_m))],
            delta: vec![Array::zeros((o_n, o_m))],
            l: 2, // Even though we have TWO layers, we're using L = 1 because we're using zero-indexing
            output: output.clone(),
            learnrate: 0.1,
            iterations: 100,
        };
        network
    }

    pub fn add_layers(mut self, layers_cfg: Vec<FCLayer>) -> Self {
        // Let's get our layer order and sizes worked out!
        self.layers_cfg = layers_cfg.clone(); // First, we'll erase the default layer information
                                              // The final layer is always based on the penultimate layer output and the required output dimensions
                                              // We'll add that now
        self.layers_cfg
            .push(FCLayer::new("sigmoid", self.output.shape()[1]));
        //println!("Network layers_cfg is now: {:#?}", self.layers_cfg);
        // Now, we'll update our `l` parameter so we know how many layers we need
        self.l = self.layers_cfg.len() + 1; // Makes sure that the `l` variable is always kept even with the layers_cfg
                                            //println!(
                                            //    "Based on that configuration, we can see the new l value is: {}",
                                            //    self.l
                                            //);
                                            // Then, we'll build a sets of z,w,a, and delta of the required sizes, to be filled afterwards
        let mut z: Vec<Array2<f64>> = vec![Array::zeros((1, 1)); self.l - 1]; // intermediate matrix products, with one less than total layers
        let mut w: Vec<Array2<f64>> = vec![Array::zeros((1, 1)); self.l - 1]; // There is one less weight matrix than total layers in the network
        let mut a: Vec<Array2<f64>> = vec![Array::zeros((1, 1)); self.l]; // output layers, where a[0] is the input matrix, so it has the length `l`
        let mut delta: Vec<Array2<f64>> = vec![Array::zeros((1, 1)); self.l - 1]; // There is one less weight matrix than total layers in the network

        a[0] = self.a[0].clone(); // The input matrix always gets the a[0] slot
        for i in 1..self.l {
            a[i] = Array::zeros((a[i - 1].shape()[0], self.layers_cfg[i - 1].output_size));
        }
        // If we've built the A matrices correct, then we can use those shapes and the information in the
        // layers_cfg vector to build the W,Z, and delta matrices
        for i in 0..(self.l - 1) {
            //w[i] = Array::zeros((a[i].shape()[1],self.layers_cfg[i].output_size));
            w[i] = Array::random(
                (a[i].shape()[1], self.layers_cfg[i].output_size),
                Uniform::new(-1., 1.),
            );
            z[i] = Array::zeros((a[i + 1].shape()[0], a[i + 1].shape()[1]));
            // let index = self.l - (i +1);
            delta[i] = Array::zeros((z[i].shape()[0], z[i].shape()[1]));
        }

        // Now that we've built a functioning system of z,w,a, and delta matrices, we'll
        // copy them over to the network's owned parameters
        self.z = z;
        self.w = w;
        self.a = a;
        self.delta = delta;
        // &self.forward_pass();
        self
    }

    pub fn backwards_pass(&mut self) {
        // Backwards pass
        // From basic net
        // WORKING BACKWARDS PASS FOR LAYER L
        // YES // self.delta[1] = (self.a[2].clone() - self.output.clone()) * self.z[1].clone().mapv(|x| sigmoid_prime(x)) * self.learnrate;

        /*let delta_l = self.calculate_error()
        * self.z[self.l - 2]
            .map(|x| activation_function_prime(&self.layers_cfg, self.l - 2, *x))
        * self.learnrate;*/

        self.delta[self.l - 2] = self.calculate_error()
            * self.z[self.l - 2]
                .map(|x| activation_function_prime(&self.layers_cfg, self.l - 2, *x))
            * self.learnrate; // This is because self.l is total layers, but we need to subtract one for both 0-indexing and beacuse of the relative number of delta matrices
                              // YES // let delta_w1 = self.a[1].t().dot(&self.delta[1]);
                              // YES // self.w[1] = self.w[1].clone() - delta_w1;
                              // YES// self.w[1] = self.w[1].clone() - self.a[1].t().dot(&self.delta[1]);
        let l_index = self.l - 2;
        self.w[l_index] = &self.w[l_index] - &self.a[l_index].t().dot(&self.delta[l_index]);

        // WORKING BACKWARDS PASS FOR LAYERS [0;l)
        // YES //self.delta[0] = self.delta[1].dot(&self.w[1].t()) * self.z[0].clone().mapv(|x| sigmoid_prime(x));
        // YES // let delta_w0 = self.a[0].t().dot(&self.delta[0]);
        // YES // self.w[0] = self.w[0].clone() - delta_w0;
        if self.l > 2 {
            // The special case is a two-layer (input -> output) network
            for i in 0..(self.l - 2) {
                let index = (self.l - 3) - i;
                //println!("i = {} -> index = {}",i,index);
                //println!("Should be assigning a delta value to self.delta[{}] ",index);
                self.delta[index] = self.delta[index + 1].dot(&self.w[index + 1].t())
                    * self.z[index].mapv(|x| activation_function_prime(&self.layers_cfg, index, x));
                //let dE_over_dW_index = self.a[index].t().dot(&self.delta[index]);
                self.w[index] = &self.w[index] - &self.a[index].t().dot(&self.delta[index]); // &dE_over_dW_index;
            }
        }
    }

    pub fn forward_pass(&mut self) {
        for i in 0..(self.l - 1) {
            //z[1] = a[0].dot(&w[0]);
            // There are l-1 z matrices, which are based on the a and w vectors from the previous layer
            self.z[i] = self.a[i].dot(&self.w[i]);
            self.a[i + 1] = self.z[i].mapv(|x| activation_function(&self.layers_cfg, i, x));
        }
    }

    pub fn train(&mut self) -> Model {
        for i in 0..self.iterations {
            self.forward_pass();
            self.backwards_pass();
            //println!("network:\n{:#?}",self.w[self.l-2]);
            println!(
                "In training iteration #{}, summed error is: {}",
                i,
                self.calculate_error().sum()
            );
            // if self.calculate_error().sum().abs() < 1.0 { break; } // Break the training loop early
        }
        Model {
            w: self.w.clone(),
            layers_cfg: self.layers_cfg.clone(),
        }
    }

    pub fn sgd_train(&mut self, group_size: usize) -> Model {
        let mut rng = thread_rng();
        for i in 0..self.iterations {
            let group_start = rng.gen_range(0, self.a[0].nrows() - group_size);

            let input = self.a[0]
                .slice(s![group_start..group_start + group_size, ..])
                .to_owned();
            let output = self
                .output
                .slice(s![group_start..group_start + group_size, ..])
                .to_owned();

            let mut sgd_network = FullyConnectedNetwork::default(input, output)
                .add_layers(
                    self.layers_cfg
                        .clone()
                        .drain(0..self.layers_cfg.len() - 1)
                        .collect(),
                )
                .build();
            sgd_network.w = self.w.clone();
            sgd_network.backwards_pass();
            // println!("sgd_network:\n{:#?}",sgd_network);
            sgd_network.forward_pass();
            sgd_network.backwards_pass();

            self.w = sgd_network.w.clone();
            self.forward_pass();

            println!(
                "In training iteration #{}, summed error is: {}",
                i,
                self.calculate_error().sum()
            );
        }

        Model {
            w: self.w.clone(),
            layers_cfg: self.layers_cfg.clone(),
        }
    }

    pub fn calculate_error(&self) -> Array2<f64> {
        let mut error = self.a.last().unwrap() - &self.output;
        error = error.map(|x| if *x >= 0. {x*x} else { (x*x) * -1.} );
        error
    }
}
