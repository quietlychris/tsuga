use crate::activation_functions::*;
use crate::fc_layer::*;
use crate::*;

#[cfg(feature = "gpu")]
use carya::opencl::*;

#[cfg(feature = "gpu")]
use ocl::Error;

use rand::prelude::*;
use std::iter::{FromIterator, Iterator};

pub fn create_vec(arr: &Array2<f32>) -> Vec<f32> {
    Array::from_iter(arr.iter().cloned()).to_vec()
}

// TO_DO: The NN fields are all currently public, but this might not be required as a final configuration
#[derive(Debug, Clone)]
pub struct FullyConnectedNetwork {
    input: Array2<f32>,  // The training data
    output: Array2<f32>, // The training target output
    layers_cfg: Vec<FCLayer>,
    z: Vec<Array2<f32>>,     // intermediate matrix products
    pub w: Vec<Array2<f32>>, // weight matrices
    a: Vec<Array2<f32>>,     // output layers
    delta: Vec<Array2<f32>>, // the delta matrix for backpropogation
    l: usize,                // number of layers in the neural network
    learnrate: f32,          // learnrate of the network, often "alpha" in equations
    iterations: usize,       // number of training iterations
    min_iterations: usize,
    error_threshold: f32,
    batch_size: usize,
}

impl FullyConnectedNetwork {
    pub fn print_shape(&self) {
        println!("layers_cfg:\n{:#?}\n", self.layers_cfg);
        println!("^ -> l = {}", self.l);
        for i in 0..self.a.len() {
            println!("a[{}].shape(): {:?}", i, self.a[i].shape());
        }
        for i in 0..self.z.len() {
            println!("z[{}].shape(): {:?}", i, self.z[i].shape());
        }
        for i in 0..self.delta.len() {
            println!("delta[{}].shape(): {:?}", i, self.delta[i].shape());
        }
        for i in 0..self.w.len() {
            println!("w[{}].shape(): {:?}", i, self.w[i].shape());
        }
    }

    pub fn learnrate(mut self, learnrate: f32) -> Self {
        self.learnrate = learnrate;
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

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn build(self) -> FullyConnectedNetwork {
        FullyConnectedNetwork {
            input: self.input,
            output: self.output,
            layers_cfg: self.layers_cfg,
            z: self.z,
            w: self.w,
            a: self.a,
            delta: self.delta,
            l: self.l,
            learnrate: self.learnrate,
            iterations: self.iterations,
            min_iterations: self.min_iterations,
            error_threshold: self.error_threshold,
            batch_size: self.batch_size,
        }
    }

    pub fn default(input: Array2<f32>, output: Array2<f32>) -> Self {
        let (o_n, o_m) = (output.nrows(), output.ncols());
        let network = FullyConnectedNetwork {
            input: input.clone(),
            output: output.clone(),
            layers_cfg: vec![FCLayer::new("sigmoid", o_m)],
            z: vec![Array::zeros((o_n, o_m))],
            w: vec![Array::random(
                (input.ncols(), output.ncols()),
                Uniform::new(-0.5, 0.5),
            )],
            a: vec![input.clone(), Array::zeros((o_n, o_m))],
            delta: vec![Array::zeros((o_n, o_m))],
            l: 2, // Even though we have TWO layers, we're using L = 1 because we're using zero-indexing
            learnrate: 0.1,
            iterations: 100,
            min_iterations: 0,
            error_threshold: 0.01,
            batch_size: 200,
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

        a[0] = self.a[0]
            .clone()
            .slice(s![0..self.batch_size, ..])
            .to_owned(); // The input matrix always gets the a[0] slot
        for i in 1..self.l {
            a[i] = Array::zeros((a[i - 1].nrows(), self.layers_cfg[i - 1].output_size));
        }
        // If we've built the A matrices correct, then we can use those shapes and the information in the
        // layers_cfg vector to build the W,Z, and delta matrices
        for i in 0..(self.l - 1) {
            w[i] = Array::random(
                (a[i].ncols(), self.layers_cfg[i].output_size),
                Uniform::new(-0.3, 0.3),
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

        self
    }

    #[inline]
    pub fn backwards_pass(&mut self, num: usize, batch_size: usize) {
        let alpha = self.l - 1; // The highest value of a.len() - 1 bc of zero indexing
        let bravo = self.l - 2;

        let error = &self.a[alpha] - &self.output.slice(s![num..num + batch_size, ..]);
        self.delta[bravo] = &error * &self.z[bravo].mapv(|x| sigmoid_prime(x)) * self.learnrate;
        let dw = &self.a[bravo].t().dot(&self.delta[bravo]);
        self.w[bravo] -= dw;

        for layer in { 0..bravo }.rev() {
            self.delta[layer] = self.delta[layer + 1].dot(&self.w[layer + 1].t())
                * self.z[layer].mapv(|x| sigmoid_prime(x));
            let dw = &self.a[layer].t().dot(&self.delta[layer]);
            self.w[layer] -= dw;
        }
    }

    #[inline]
    pub fn forward_pass(&mut self, num: usize, batch_size: usize) {
        if (num + batch_size) <= self.input.nrows() {
            self.a[0] = self.input.slice(s![num..num + batch_size, ..]).to_owned();
            for layer in 0..=(self.l - 2) {
                self.z[layer] = self.a[layer].dot(&self.w[layer]);
                self.a[layer + 1] = self.z[layer].clone().mapv(|x| sigmoid(x));
            }
        } else {
            panic!("Forward pass operation has invalid array sizes");
        }
    }

    #[inline]
    pub fn train(&mut self) {
        let mut rng = rand::thread_rng();
        let mut num: usize;

        for iteration in 0..self.iterations {
            num = rng.gen_range(0, self.input.nrows() - self.batch_size);
            self.forward_pass(num, self.batch_size);
            self.backwards_pass(num, self.batch_size);
            let error =
                &self.a[self.l - 1] - &self.output.slice(s![num..num + self.batch_size, ..]);
            println!(
                "Training iteration #{}, % error: {}",
                iteration,
                error.sum().abs() / self.batch_size as f32
            );
        }
    }

    pub fn evaluate(mut self, input: Array2<f32>) -> Array2<f32> {
        self.input = input.clone();
        self.a[0] = input.clone();
        self.forward_pass(0, 10_000);
        self.a.last().unwrap().clone()
    }

    #[inline]
    #[cfg(feature = "gpu")]
    pub fn train_w_carya(&mut self, gpu_choice: &str) -> Result<(), Error> {
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
        let mut z: Vec<OpenCLArray> = Vec::with_capacity(self.z.len());
        for i in 0..self.z.len() {
            z.push(OpenCLArray::from_array(backend.clone(), &self.z[i])?);
        }
        let output: OpenCLArray = OpenCLArray::from_array(backend.clone(), &self.output)?;

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

            // let mut temp_delta = OpenCLArray::new(backend.clone(),delta[l_index].rows,delta[l_index].cols)?;
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
        for i in 0..self.z.len() {
            self.z[i] = z[i].clone().to_array()?;
        }
        Ok(())
    }
}
