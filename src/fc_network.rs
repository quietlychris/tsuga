use crate::activation_functions::*;
use crate::fc_layer::*;
use crate::fc_model::*;
use crate::*;

#[cfg(feature = "gpu")]
use carya::opencl::*;

use image::*;

#[cfg(feature = "gpu")]
use ocl::Error;

use rand::prelude::*;
use std::iter::{FromIterator, Iterator};
use std::time::{Duration, Instant};
use ndarray::stack;

pub fn create_vec(arr: &Array2<f32>) -> Vec<f32> {
    Array::from_iter(arr.iter().cloned()).to_vec()
}

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
    pub error_threshold: f32,
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

    pub fn error_threshold(mut self, error_threshold: f32) -> Self {
        self.error_threshold = error_threshold;
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
            error_threshold: self.error_threshold,
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
            error_threshold: 1.0,
        };
        network
    }

    #[inline]
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

    #[inline]
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
        self.delta[l_index] = &self.calculate_error()
            * &self.z[l_index].map(|x| activation_function_prime(&self.layers_cfg, l_index, *x))
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
            println!("l = {}",self.l);
            for i in 0..(self.l - 2) {
                let index = (self.l - 3) - i;
                //println!("i = {} -> index = {}",i,index);
                //println!("Should be assigning a delta value to self.delta[{}] ",index);
                self.delta[index] = &self.delta[index + 1].dot(&self.w[index + 1].t())
                    * &self.z[index].mapv(|x| activation_function_prime(&self.layers_cfg, index, x));
                self.b[index] =
                    &self.b[index] + &self.delta[index].map(|x| x * -self.bias_learnrate);
                //let dE_over_dW_index = self.a[index].t().dot(&self.delta[index]);
                self.w[index] = &self.w[index] - &self.a[index].t().dot(&self.delta[index]);
                // &dE_over_dW_index;
            }
        }
    }

    #[inline]
    pub fn forward_pass(&mut self) {
        for i in 0..(self.l - 1) {
            //z[1] = a[0].dot(&w[0]);
            // There are l-1 z matrices, which are based on the a and w vectors from the previous layer
            self.z[i] = self.a[i].dot(&self.w[i]);
            self.a[i + 1] =
                &self.z[i].mapv(|x| activation_function(&self.layers_cfg, i, x)) + &self.b[i];
        }
        softmax(&mut self.a[self.l-1]);
    }

    #[inline]
    pub fn train(&mut self) -> Model {
        for iteration in 0..self.iterations {
            self.forward_pass();
            self.backwards_pass();
            //println!("network:\n{:#?}",self.w[self.l-2]);
            let sum_error = self.calculate_error().sum();
            println!(
                "In training iteration #{}, summed error is: {}",
                iteration, sum_error
            );
            if iteration > self.min_iterations {
                if sum_error.abs() < self.error_threshold.abs() {
                    break;
                }
            }
        }
        Model {
            w: self.w.clone(),
            layers_cfg: self.layers_cfg.clone(),
        }
    }

    #[inline]
    #[cfg(feature = "gpu")]
    pub fn train_w_carya(&mut self, gpu_choice: &str) -> Result<Model, Error> {
        let backend = CLBackEnd::new(gpu_choice)?;
        let mut a: Vec<OpenCLArray> = Vec::with_capacity(self.a.len());
        for i in 0..self.a.len() {
            a.push(OpenCLArray::from_array(backend.clone(), &self.a[i])?);
        }
        let mut w: Vec<OpenCLArray> = Vec::with_capacity(self.w.len());
        for i in 0..self.w.len() {
            w.push(OpenCLArray::from_array(backend.clone(), &self.w[i])?);
            // println!("w[{}] = {:?}",i,w[i].clone().to_array()?);
        }
        let mut delta: Vec<OpenCLArray> = Vec::with_capacity(self.delta.len());
        for i in 0..self.delta.len() {
            delta.push(OpenCLArray::from_array(backend.clone(), &self.delta[i])?);
        }
        let mut b: Vec<OpenCLArray> = Vec::with_capacity(self.b.len());
        for i in 0..self.b.len() {
            b.push(OpenCLArray::from_array(backend.clone(), &self.b[i])?);
        }
        let mut z: Vec<OpenCLArray> = Vec::with_capacity(self.z.len());
        for i in 0..self.z.len() {
            z.push(OpenCLArray::from_array(backend.clone(), &self.z[i])?);
        }
        let mut output: OpenCLArray = OpenCLArray::from_array(backend.clone(), &self.output)?;

        // Intermediate products
        let mut error = OpenCLArray::new(backend.clone(), output.rows, output.cols)?;
        let mut temp_a: Vec<OpenCLArray> = Vec::with_capacity(self.a.len());
        for i in 0..self.a.len() {
            temp_a.push(OpenCLArray::from_array(backend.clone(), &self.a[i])?);
        }
        let mut temp_w: Vec<OpenCLArray> = Vec::with_capacity(self.w.len());
        for i in 0..self.w.len() {
            temp_w.push(OpenCLArray::from_array(backend.clone(), &self.w[i])?);
            // println!("w[{}] = {:?}",i,w[i].clone().to_array()?);
        }
        let mut temp_delta: Vec<OpenCLArray> = Vec::with_capacity(self.delta.len());
        for i in 0..self.delta.len() {
            temp_delta.push(OpenCLArray::from_array(backend.clone(), &self.delta[i])?);
        }
        let mut temp_b: Vec<OpenCLArray> = Vec::with_capacity(self.b.len());
        for i in 0..self.b.len() {
            temp_b.push(OpenCLArray::from_array(backend.clone(), &self.b[i])?);
        }
        let mut temp_z: Vec<OpenCLArray> = Vec::with_capacity(self.z.len());
        for i in 0..self.z.len() {
            temp_z.push(OpenCLArray::from_array(backend.clone(), &self.z[i])?);
        }

        let start = Instant::now();
        for iteration in 0..self.iterations {
            // Forward pass------------------------------------------------------------------------------
            for i in 0..(self.l - 1) {
                //z[1] = a[0].dot(&w[0]);
                // There are l-1 z matrices, which are based on the a and w vectors from the previous layer
                // self.z[i] = self.a[i].dot(&self.w[i]);
                a[i].dot(&w[i], &mut z[i])?; // Carya

                // self.a[i + 1] =
                //     self.z[i].mapv(|x| activation_function(&self.layers_cfg, i, x)) + &self.b[i];
                z[i].sigmoid(&mut a[i + 1])?;
                &a[i + 1].clone().add(&b[i], &mut a[i + 1])?;
            }
            println!(
                "Iteration {}, End of forward pass: {:?} s",
                iteration,
                start.elapsed().as_secs()
            );
            // Backwards pass-----------------------------------------------------------------------------
            let l_index = self.l - 2;

            a.last().unwrap().subtract(&output, &mut error)?;
            temp_z[l_index] = z[l_index].clone();
            temp_z[l_index].sigmoid_prime(&mut z[l_index])?;
            error.hadamard(&z[l_index], &mut delta[l_index])?;

            println!(
                "In training iteration #{}, summed error is: {}",
                iteration,
                error.clone().to_array()?.sum()
            );
            temp_delta[l_index] = delta[l_index].clone();
            temp_delta[l_index].scalar_multiply(self.learnrate, &mut delta[l_index])?;

            // self.w[l_index] = &self.w[l_index] - &self.a[l_index].t().dot(&self.delta[l_index]);

            //let mut temp = OpenCLArray::new(backend.clone(),a[l_index].cols,delta[l_index].cols)?;
            temp_a[l_index].rows = a[l_index].cols;
            temp_a[l_index].cols = delta[l_index].cols;
            a[l_index].t_v2()?;
            a[l_index].dot(&delta[l_index], &mut temp_a[l_index])?;
            a[l_index].t_v2()?;

            // println!("w[l_index] before subtraction =\n{:#?}",w[l_index].clone().to_array()?);
            w[l_index]
                .clone()
                .subtract(&temp_a[l_index], &mut w[l_index])?;
            // println!("w[l_index] after subtraction =\n{:#?}",w[l_index].clone().to_array()?);

            // self.b[l_index] =
            //     &self.b[l_index] + &self.delta[l_index].map(|x| *x * -self.bias_learnrate);

            // let mut temp_delta = OpenCLArray::new(backend.clone(),delta[l_index].rows,delta[l_index].cols)?;
            delta[l_index].scalar_multiply(-self.bias_learnrate, &mut temp_delta[l_index])?;
            b[l_index]
                .clone()
                .add(&temp_delta[l_index], &mut b[l_index])?;
            println!(
                "Iteration {}, end of first layer of backward pass: {:?} s",
                iteration,
                start.elapsed().as_secs()
            );
            if self.l > 2 {
                // The special case is a two-layer (input -> output) network
                for i in 0..(self.l - 2) {
                    let index = (self.l - 3) - i;

                    // self.delta[index] = self.delta[index + 1].dot(&self.w[index + 1].t())
                    //     * self.z[index].mapv(|x| activation_function_prime(&self.layers_cfg, index, x));

                    // let mut temp1 = OpenCLArray::new(backend.clone(),delta[index].rows,delta[index].cols)?;
                    w[index + 1].t_v2()?;
                    delta[index + 1].dot(&w[index + 1], &mut temp_delta[index])?;
                    w[index + 1].t_v2()?;
                    z[index].sigmoid_prime(&mut temp_z[index])?;
                    temp_delta[index].hadamard(&temp_z[index], &mut delta[index])?;

                    // self.b[index] =
                    //     &self.b[index] + &self.delta[index].map(|x| x * -self.bias_learnrate);

                    // let mut temp2 = OpenCLArray::new(backend.clone(),b[index].rows,b[index].cols)?;

                    delta[index].scalar_multiply(-self.bias_learnrate, &mut temp_b[index])?;
                    b[index].clone().add(&temp_b[index], &mut b[index])?;

                    //let dE_over_dW_index = self.a[index].t().dot(&self.delta[index]);
                    //self.w[index] = &self.w[index] - &self.a[index].t().dot(&self.delta[index]);

                    // let mut temp3 = OpenCLArray::new(backend.clone(),w[index].rows,w[index].cols)?;
                    a[index].t_v2()?;
                    a[index].dot(&delta[index], &mut temp_w[index])?;
                    a[index].t_v2()?;
                    w[index].clone().subtract(&temp_w[index], &mut w[index])?;
                }
            }
        }
        println!("End of backward pass: {:?} s", start.elapsed().as_secs());
        // Write the OpenCL result vectors back to the original ndarray matrices
        for i in 0..self.a.len() {
            self.a[i] = a[i].clone().to_array()?;
        }
        for i in 0..self.w.len() {
            self.w[i] = w[i].clone().to_array()?;
        }
        for i in 0..self.delta.len() {
            self.delta[i] = delta[i].clone().to_array()?;
        }
        for i in 0..self.b.len() {
            self.b[i] = b[i].clone().to_array()?;
        }
        for i in 0..self.z.len() {
            self.z[i] = z[i].clone().to_array()?;
        }

        Ok(Model {
            w: self.w.clone(),
            layers_cfg: self.layers_cfg.clone(),
        })
    }

    #[inline]
    pub fn sgd_train(&mut self, batch_size: usize) -> Model {
        let mut rng = thread_rng();

        let input_cols = self.a[0].ncols();
        let output_cols = self.output.ncols();

        let mut input = self.a[0].slice(s![0..=batch_size, ..]).to_owned();
        let mut output = self.output.slice(s![0..=batch_size, ..]).to_owned();
        let mut sgd_network = FullyConnectedNetwork::default(input.clone(), output.clone())
            .add_layers(
                self.layers_cfg
                    .clone()
                    .drain(0..self.layers_cfg.len() - 1)
                    .collect(),
            )
            .iterations(1)
            .build();

        let mut group = vec![0; batch_size];
        for i in 0..self.iterations {
            for i in 0..batch_size {
                group[i] = rng.gen_range(0, self.a[0].nrows());
            }

            let mut input: Array2<f32> = self.a[0]
                .slice(s![group[0], ..])
                .to_owned()
                .into_shape((1, input_cols))
                .unwrap();
            for record in &group {
                let intermediate: Array2<f32> = self.a[0]
                    .slice(s![*record, ..])
                    .to_owned()
                    .into_shape((1, input_cols))
                    .unwrap();
                input = stack![Axis(0), input.clone(), intermediate];
            }

            let mut output: Array2<f32> = self.output
                .slice(s![group[0], ..])
                // .clone()
                .to_owned()
                .into_shape((1, output_cols))
                .unwrap();
            for record in &group {
                let intermediate: Array2<f32> = self.output
                    .slice(s![*record, ..])
                    .to_owned()
                    .into_shape((1, output_cols))
                    .unwrap();
                output = stack![Axis(0), output.clone(), intermediate];
            }

            sgd_network.a[0] = input;
            sgd_network.output = output;

            sgd_network.forward_pass();
            sgd_network.backwards_pass();


            println!(
                "In training iteration #{}, summed error is: {}",
                i,
                sgd_network.calculate_error().sum()
            );
        }

        self.w = sgd_network.w;

        Model {
            w: self.w.clone(),
            layers_cfg: self.layers_cfg.clone(),
        }
    }

    #[inline]
    pub fn calculate_error(&self) -> Array2<f32> {
        let mut error = *&self.a.last().unwrap() - &self.output;
        error = error.map(|x| if *x >= 0. { x * x } else { (x * x) * -1. });
        error
    }
}
