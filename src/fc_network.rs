use crate::activation_functions::*;
use crate::fc_layer::*;
use crate::fc_model::*;
use crate::linalg_ocl::*;
use crate::*;

use image::*;
use rand::prelude::*;
use std::iter::{Iterator,FromIterator};

// TO_DO: The NN fields are all currently public, but this might not be required as a final configuration
#[derive(Debug, Clone)]
pub struct FullyConnectedNetwork {
    pub layers_cfg: Vec<FCLayer>,
    pub z: Vec<Array2<f32>>,     // intermediate matrix products
    pub w: Vec<Array2<f32>>,     // weight matrices
    pub a: Vec<Array2<f32>>,     // output layers
    pub delta: Vec<Array2<f32>>, // the delta matrix for backpropogation
    pub b: Vec<Array2<f32>>,     // the bias matrix
    pub output: Array2<f32>,     // The target output layer
    pub l: usize,                // number of layers in the neural network
    pub learnrate: f32,          // learnrate of the network, often "alpha" in equations
    pub bias_learnrate: f32,
    pub iterations: usize, // number of training iterations
    pub min_iterations: usize,
}

impl FullyConnectedNetwork {
    pub fn print_shape(&self) {
        for i in 0..self.a.len() {
            println!("a[{}].shape(): {:?}", i, self.a[i].shape());
        }
        for i in 0..self.z.len() {
            println!("z[{}].shape(): {:?}", i, self.z[i].shape());
        }
        for i in 0..self.b.len() {
            println!("b[{}].shape(): {:?}", i, self.b[i].shape());
        }
        for i in 0..self.delta.len() {
            println!("delta[{}].shape(): {:?}", i, self.delta[i].shape());
        }
        for i in 0..self.w.len() {
            println!("w[{}].shape(): {:?}", i, self.w[i].shape());
        }
    }

    pub fn update_weights(&mut self, w: Vec<Array2<f32>>) {
        self.w = w;
    }

    pub fn learnrate(mut self, learnrate: f32) -> Self {
        self.learnrate = learnrate;
        self
    }

    pub fn bias_learnrate(mut self, bias_learnrate: f32) -> Self {
        self.bias_learnrate = bias_learnrate;
        self
    }

    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn min_iterations(mut self, min_iterations: usize) -> Self {
        self.min_iterations = min_iterations;
        self
    }

    pub fn build(self) -> FullyConnectedNetwork {
        FullyConnectedNetwork {
            layers_cfg: self.layers_cfg,
            z: self.z,
            w: self.w,
            a: self.a,
            delta: self.delta,
            b: self.b,
            output: self.output,
            l: self.l,
            learnrate: self.learnrate,
            bias_learnrate: self.bias_learnrate,
            iterations: self.iterations,
            min_iterations: self.min_iterations,
        }
    }

    pub fn default(input: Array2<f32>, output: Array2<f32>) -> Self {
        let (o_n, o_m) = (output.nrows(), output.ncols());
        let network = FullyConnectedNetwork {
            layers_cfg: vec![FCLayer::new("sigmoid", o_m)],
            z: vec![Array::zeros((o_n, o_m))],
            w: vec![Array::random(
                (input.ncols(), output.ncols()),
                Uniform::new(-0.5, 0.5),
            )],
            a: vec![input.clone(), Array::zeros((o_n, o_m))],
            delta: vec![Array::zeros((o_n, o_m))],
            b: vec![Array::zeros((o_n, o_m))],
            l: 2, // Even though we have TWO layers, we're using L = 1 because we're using zero-indexing
            output: output.clone(),
            learnrate: 0.1,
            bias_learnrate: 0.01,
            iterations: 100,
            min_iterations: 0,
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
        let mut z: Vec<Array2<f32>> = vec![Array::zeros((1, 1)); self.l - 1]; // intermediate matrix products, with one less than total layers
        let mut w: Vec<Array2<f32>> = vec![Array::zeros((1, 1)); self.l - 1]; // There is one less weight matrix than total layers in the network
        let mut a: Vec<Array2<f32>> = vec![Array::zeros((1, 1)); self.l]; // output layers, where a[0] is the input matrix, so it has the length `l`
        let mut delta: Vec<Array2<f32>> = vec![Array::zeros((1, 1)); self.l - 1]; // There is one less weight matrix than total layers in the network
        let mut b: Vec<Array2<f32>> = vec![Array::zeros((1, 1)); self.l - 1]; // There is one less weight matrix than total layers in the network

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
            b[i] = Array::zeros((z[i].shape()[0], z[i].shape()[1]));
        }

        // Now that we've built a functioning system of z,w,a, and delta matrices, we'll
        // copy them over to the network's owned parameters
        self.z = z;
        self.w = w;
        self.a = a;
        self.delta = delta;
        self.b = b;
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

        let l_index = self.l - 2;
        self.delta[l_index] = self.calculate_error()
            * self.z[l_index].map(|x| activation_function_prime(&self.layers_cfg, l_index, *x))
            * self.learnrate;

        // This is because self.l is total layers, but we need to subtract one for both 0-indexing and beacuse of the relative number of delta matrices
        // YES // let delta_w1 = self.a[1].t().dot(&self.delta[1]);
        // YES // self.w[1] = self.w[1].clone() - delta_w1;
        // YES// self.w[1] = self.w[1].clone() - self.a[1].t().dot(&self.delta[1]);

        self.w[l_index] = &self.w[l_index] - &self.a[l_index].t().dot(&self.delta[l_index]);
        self.b[l_index] =
            &self.b[l_index] + &self.delta[l_index].map(|x| *x * -self.bias_learnrate);

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
                self.b[index] =
                    &self.b[index] + &self.delta[index].map(|x| x * -self.bias_learnrate);
                //let dE_over_dW_index = self.a[index].t().dot(&self.delta[index]);
                self.w[index] = &self.w[index] - &self.a[index].t().dot(&self.delta[index]);
                // &dE_over_dW_index;
            }
        }
    }

    pub fn forward_pass(&mut self) {
        for i in 0..(self.l - 1) {
            //z[1] = a[0].dot(&w[0]);
            // There are l-1 z matrices, which are based on the a and w vectors from the previous layer
            self.z[i] = self.a[i].dot(&self.w[i]);
            self.a[i + 1] =
                self.z[i].mapv(|x| activation_function(&self.layers_cfg, i, x)) + &self.b[i];
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
            if self.calculate_error().sum().abs() < 100. && i > self.min_iterations {
                break;
            }
            // if self.calculate_error().sum().abs() < 1.0 { break; } // Break the training loop early
        }
        Model {
            w: self.w.clone(),
            layers_cfg: self.layers_cfg.clone(),
        }
    }

    pub fn train_on_gpu(&mut self, gpu_choice: &str) -> Model {
        // Convert the global vectors to our local arrays
        // println!("self.a: {:?}",self.a);
        let mut a: Vec<Vec<f32>> = Vec::with_capacity(self.a.len());
        for i in 0..self.a.len() {
            a.push(Array::from_iter(self.a[i].iter().cloned()).to_vec());
        }
        // println!("a: {:?}",a);
        let mut w: Vec<Vec<f32>> = Vec::with_capacity(self.w.len());
        for i in 0..self.w.len() {
            w.push(Array::from_iter(self.w[i].iter().cloned()).to_vec());
        }
        let mut delta: Vec<Vec<f32>> = Vec::with_capacity(self.delta.len());
        for i in 0..self.delta.len() {
            delta.push(Array::from_iter(self.delta[i].iter().cloned()).to_vec());
        }
        let mut b: Vec<Vec<f32>> = Vec::with_capacity(self.b.len());
        for i in 0..self.b.len() {
            b.push(Array::from_iter(self.b[i].iter().cloned()).to_vec());
        }
        let mut z: Vec<Vec<f32>> = Vec::with_capacity(self.z.len());
        for i in 0..self.z.len() {
            z.push(Array::from_iter(self.z[i].iter().cloned()).to_vec());
        }
        let mut output: Vec<f32> = Array::from_iter(self.output.iter().cloned()).to_vec();

        // Training iterations
        let mut ctx: ocl::ProQue = build_ocl_proque(gpu_choice.to_string());

        for iteration in 0..self.iterations {
            // FORWARD PASS
            for i in 0..(self.l - 1) {
                z[i] = dot_product(
                    &mut ctx,
                    &a[i],
                    &w[i],
                    (self.a[i].nrows(), self.a[i].ncols(), self.w[i].ncols()),
                )
                .expect(
                    "Couldn't run the OpenCL dot product operation on the a[i] and w[i] matrices",
                );
                // println!("z[{}]:\n{:#?}", i, z[i]);
                a[i+1] = linalg_ocl::sigmoid(&mut ctx, &z[i],(self.z[i].nrows(), self.z[i].ncols()))
                    .expect("Couldn't run the OpenCL sigmoid operations on z[i] matrix while assigning to a[i+1]");
            }

            // ----------- BACKWARDS PASS --------------------------
            let l_index = self.l - 2;
            // self.delta[l_index] = self.calculate_error()
            //     * self.z[l_index].map(|x| activation_function_prime(&self.layers_cfg, l_index, *x))
            //     * self.learnrate;
            // TO_DO: This way of doing the error calculation isn't working
            let mut error = subtract(
                &mut ctx,
                &a.last().unwrap(),
                &output,
                (
                    self.a.last().unwrap().nrows(),
                    self.a.last().unwrap().ncols(),
                ),
            )
            .expect("Couldn't calculate the error in OpenCL");
            error = error.iter().map(|x| if *x >= 0. { x.powf(4.0) } else { (x.powf(4.0)) * -1. }).collect();

            println!("In training iteration #{}, summed error is: {}",iteration,&error.clone().iter().fold(0., |acc, x| acc + x));
            // println!("error is:\n{:#?}", error);
            
            // println!("z before sigmoid:\n{:#?}", z[l_index]);
            let applied_sigmoid_z = linalg_ocl::sigmoid(
                &mut ctx,
                &z[l_index],
                (self.z[l_index].nrows(), self.z[l_index].ncols()),
            )
            .expect("Couldn't apply the sigmoid operation");
            // println!("z after sigmoid:\n{:#?}", applied_sigmoid_z);

            let error_times_sigmoid = hadamard(
                &mut ctx,
                &error,
                &applied_sigmoid_z,
                (self.z[l_index].nrows(), self.z[l_index].ncols()),
            )
            .expect("Couldn't multiply the error by the sigmoid-applied z value");

            delta[l_index] = multiply_by_scalar(&mut ctx, &error_times_sigmoid, self.learnrate)
                .expect("Couldn't multiply the result by the learnrate");

            //-----------
            // self.w[l_index] = &self.w[l_index] - &self.a[l_index].t().dot(&self.delta[l_index]);
            let a_l_index_transposed = transpose(
                &mut ctx,
                &a[l_index],
                (self.a[l_index].nrows(), self.a[l_index].ncols()),
            )
            .unwrap();

            let (n, m, k) = (
                self.a[l_index].nrows(),
                self.delta[l_index].nrows(),
                self.delta[l_index].ncols(),
            );

            let a_t_dot_delta = dot_product(
                &mut ctx,
                &a_l_index_transposed,
                &delta[l_index],
                (
                    self.a[l_index].ncols(),
                    self.a[l_index].nrows(),
                    self.delta[l_index].ncols(),
                ),
            )
            .expect("Couldn't run the dot product operation");

            w[l_index] = subtract(
                &mut ctx,
                &w[l_index],
                &a_t_dot_delta,
                (self.w[l_index].nrows(), self.w[l_index].ncols()),
            )
            .unwrap();
            //println!("iteration {}, w:\n{:#?}",i,w);

            //-----------------------
            // self.b[l_index] =
            //     &self.b[l_index] + &self.delta[l_index].map(|x| *x * -self.bias_learnrate);

            let delta_times_bias_learnrate =
                multiply_by_scalar(&mut ctx, &delta[l_index], -self.bias_learnrate)
                    .expect("Multiplies delta by the bias learnrate");
            b[l_index] = linalg_ocl::add(
                &mut ctx,
                &b[l_index],
                &delta_times_bias_learnrate,
                (self.b[l_index].nrows(), self.b[l_index].ncols()),
            )
            .expect("Couldn't update the initial bias value");

            // WORKING BACKWARDS PASS FOR LAYERS [0;l)
            // YES //self.delta[0] = self.delta[1].dot(&self.w[1].t()) * self.z[0].clone().mapv(|x| sigmoid_prime(x));
            // YES // let delta_w0 = self.a[0].t().dot(&self.delta[0]);
            // YES // self.w[0] = self.w[0].clone() - delta_w0;

            if self.l > 2 {
                // The special case is a two-layer (input -> output) network
                for i in 0..(self.l - 2) {
                    let index = (self.l - 3) - i;

                    // self.delta[index] = self.delta[index + 1].dot(&self.w[index + 1].t())
                    // * self.z[index].mapv(|x| activation_function_prime(&self.layers_cfg, index, x));
                    let applied_sigmoid_z = linalg_ocl::sigmoid(
                        &mut ctx,
                        &z[index],
                        (self.z[index].nrows(), self.z[index].ncols()),
                    )
                    .expect("Couldn't apply the sigmoid operation");

                    let w_index_plus_one_t = transpose(
                        &mut ctx,
                        &w[index+1],
                        (self.w[index + 1].nrows(), self.w[index + 1].ncols()),
                        //(self.w[index + 1].nrows(), self.w[index + 1].ncols()),
                    )
                    .unwrap();

                    let delta_dot_w = dot_product(
                        &mut ctx,
                        &delta[index + 1],
                        &w_index_plus_one_t,
                        (
                            self.delta[index + 1].nrows(),
                            self.delta[index + 1].ncols(),
                            self.w[index + 1].nrows(),
                        ),
                    )
                    .unwrap();

                    delta[index] = hadamard(
                        &mut ctx,
                        &delta_dot_w,
                        &applied_sigmoid_z,
                        (self.z[index].nrows(), self.z[index].ncols()),
                    )
                    .unwrap();
                    
                    // self.b[index] =
                    //     &self.b[index] + &self.delta[index].map(|x| x * -self.bias_learnrate);

                    let delta_times_bias_learnrate =
                    multiply_by_scalar(&mut ctx, &delta[index], -self.bias_learnrate)
                        .expect("Multiplies delta by the bias learnrate");
                    
                        b[index] = linalg_ocl::add(
                        &mut ctx,
                        &b[index],
                        &delta_times_bias_learnrate,
                        (self.b[index].nrows(), self.b[index].ncols()),
                    )
                    .expect("Couldn't update the initial bias value");


                    // self.w[index] = &self.w[index] - &self.a[index].t().dot(&self.delta[index]);
                    let a_index_transposed = transpose(
                        &mut ctx,
                        &a[index],
                        (self.a[index].nrows(), self.a[index].ncols()),
                    )
                    .unwrap();
        
                    let (n, m, k) = (
                        self.a[index].nrows(),
                        self.delta[index].nrows(),
                        self.delta[index].ncols(),
                    );
        
                    let a_t_dot_delta = dot_product(
                        &mut ctx,
                        &a_index_transposed,
                        &delta[index],
                        (
                            self.a[index].ncols(),
                            self.a[index].nrows(),
                            self.delta[index].ncols(),
                        ),
                    )
                    .expect("Couldn't run the dot product operation");
        
                    w[index] = subtract(
                        &mut ctx,
                        &w[index],
                        &a_t_dot_delta,
                        (self.w[index].nrows(), self.w[index].ncols()),
                    )
                    .unwrap();

                }
            }
        }
        // Write the OpenCL result vectors back to the original ndarray matrices
        for i in 0..self.a.len() {
            self.a[i] = Array::from_shape_vec(self.a[i].dim(), a[i].clone())
                .expect("Coudn't convert result to properly sized array");
        }
        for i in 0..self.w.len() {
            self.w[i] = Array::from_shape_vec(self.w[i].dim(), w[i].clone())
                .expect("Coudn't convert result to properly sized array");
        }
        for i in 0..self.delta.len() {
            self.delta[i] = Array::from_shape_vec(self.delta[i].dim(), delta[i].clone())
                .expect("Coudn't convert result to properly sized array");
        }
        for i in 0..self.b.len() {
            self.b[i] = Array::from_shape_vec(self.b[i].dim(), b[i].clone())
                .expect("Coudn't convert result to properly sized array");
        }
        for i in 0..self.z.len() {
            self.z[i] = Array::from_shape_vec(self.z[i].dim(), z[i].clone())
                .expect("Coudn't convert result to properly sized array");
        }

        Model {
            w: self.w.clone(),
            layers_cfg: self.layers_cfg.clone(),
        }
    } // LAST LINE OF FUNCTION

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

    pub fn calculate_error(&self) -> Array2<f32> {
        let mut error = self.a.last().unwrap() - &self.output;
        error = error.map(|x| if *x >= 0. { x * x } else { (x * x) * -1. });
        error
    }
}
