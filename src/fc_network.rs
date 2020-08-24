use crate::activation_functions::*;
use crate::fc_layer::*;
use crate::*;

use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::*;
use std::iter::Iterator;

use std::error::Error;
use std::time::Duration;

use crossterm::event::{poll, read, Event, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use std::io::stdout;

/// A fully-connected neural network
#[derive(Debug, Clone)]
pub struct FullyConnectedNetwork {
    input: Array2<f32>,       // The training data
    output: Array2<f32>,      // The training target output
    layers_cfg: Vec<FCLayer>, // configuration data used to build the network architecture
    z: Vec<Array2<f32>>,      // intermediate matrix products
    pub w: Vec<Array2<f32>>,  // weight matrices
    a: Vec<Array2<f32>>,      // output layers
    delta: Vec<Array2<f32>>,  // the delta matrix for backpropogation
    l: usize,                 // number of layers in the neural network
    learnrate: f32,           // learnrate of the network, often "alpha" in equations
    iterations: usize,        // number of training iterations
    min_iterations: usize,    // minimum number of training iterations
    error_threshold: f32,     // threshold at which to stop training
    validation_pct: f32,      // percentage of the training data to hold back as a validation set
    batch_size: usize,        // size of the training batch
}

impl FullyConnectedNetwork {
    /// Prints the primary shape parameters of the fully-connected network. This is useful for debugging
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

    /// A hyperparameter of defining the learning rate of the network
    pub fn learnrate(mut self, learnrate: f32) -> Self {
        self.learnrate = learnrate;
        self
    }

    /// A hyperparameter number of training iterations for the network (this combined with the `batch_size` parameter can be used to set a particular number of epochs, in which every training record is seen a certain number of times)
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// A hyperparameter setting the minimum number of iterations the network must train for before returning (can be used to override the error threshold)
    pub fn min_iterations(mut self, min_iterations: usize) -> Self {
        self.min_iterations = min_iterations;
        self
    }

    /// A hyperparameter that will exit the
    pub fn error_threshold(mut self, error_threshold: f32) -> Self {
        self.error_threshold = error_threshold;
        self
    }

    /// A hyperparameter defining the size of the training batches (can be used with `iterations` to define the number of epochs)
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// A hyperparameter defining the percentage of the training data to hold back as a validation set
    pub fn validation_pct(mut self, validation_pct: f32) -> Self {
        self.validation_pct = validation_pct;
        self
    }

    /// Returns a struct with the completed network architecture
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
            validation_pct: self.validation_pct,
            batch_size: self.batch_size,
        }
    }

    /// Builds a basic two-layer network, with the default values for all hyperparameters
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
            l: 2, // Remember, we're zero-indexing
            learnrate: 0.1,
            iterations: 100,
            min_iterations: usize::MAX,
            error_threshold: 0.,
            validation_pct: 0.2,
            batch_size: 200,
        };
        network
    }

    /// Takes a Vec<FCLayer> configuration and modifies the network architecture to include properly-sized layers with activiation functions
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

    /// Runs backpropagation on the fully-connected network, using the differivatives of each layer's activation function. The delta rule is used.
    #[inline]
    fn backwards_pass(&mut self, num: usize, batch_size: usize) {
        let alpha = self.l - 1; // The highest value of a.len() - 1 bc of zero indexing
        let bravo = self.l - 2;

        let error = &self.a[alpha] - &self.output.slice(s![num..num + batch_size, ..]);
        // Matching the proper activation function
        let activation_fn = self.layers_cfg[bravo].activation_function.as_str();
        match activation_fn {
            "sigmoid" => {
                // Single-threaded
                // self.delta[bravo] =
                //     &error * &self.z[bravo].mapv(|x| sigmoid_prime(x)) * self.learnrate

                // Multi-threaded
                self.z[bravo].par_mapv_inplace(|x| sigmoid_prime(x));
                self.delta[bravo] = &error * &self.z[bravo] * self.learnrate
            }
            "relu" => {
                // Single-threaded
                // self.delta[bravo] = &error * &self.z[bravo].mapv(|x| relu_prime(x)) * self.learnrate

                // Multi-threaded
                self.z[bravo].par_mapv_inplace(|x| relu_prime(x));
                self.delta[bravo] = &error * &self.z[bravo] * self.learnrate
            }
            _ => panic!(format!(
                "This activation function ({}) is not supported",
                activation_fn
            )),
        }

        let dw = &self.a[bravo].t().dot(&self.delta[bravo]);
        self.w[bravo] -= dw;

        for layer in { 0..bravo }.rev() {
            let activation_fn = self.layers_cfg[layer].activation_function.as_str();
            match activation_fn {
                "sigmoid" => {
                    // Single-threaded
                    // self.delta[layer] = self.delta[layer + 1].dot(&self.w[layer + 1].t())
                    //     * self.z[layer].mapv(|x| sigmoid_prime(x))

                    // Multi-threaded
                    self.z[layer].par_mapv_inplace(|x| sigmoid_prime(x));
                    self.delta[layer] =
                        self.delta[layer + 1].dot(&self.w[layer + 1].t()) * &self.z[layer]
                }
                "relu" => {
                    // Single-threaded
                    // self.delta[layer] = self.delta[layer + 1].dot(&self.w[layer + 1].t())
                    //     * self.z[layer].mapv(|x| relu_prime(x))

                    // Multi-threaded
                    self.z[layer].par_mapv_inplace(|x| relu_prime(x));
                    self.delta[layer] =
                        self.delta[layer + 1].dot(&self.w[layer + 1].t()) * &self.z[layer]
                }
                _ => panic!(format!(
                    "This activation function ({}) is not supported",
                    activation_fn
                )),
            }

            let dw = &self.a[layer].t().dot(&self.delta[layer]);
            self.w[layer] -= dw;
        }
    }

    /// Runs a forward pass on the fully-connected network, including each layer's activation function
    #[inline]
    fn forward_pass(&mut self, num: usize, batch_size: usize) {
        if (num + batch_size) <= self.input.nrows() {
            self.a[0] = self.input.slice(s![num..num + batch_size, ..]).to_owned();
            for layer in 0..=(self.l - 2) {
                self.z[layer] = self.a[layer].dot(&self.w[layer]);
                let activation_fn = self.layers_cfg[layer].activation_function.as_str();
                match activation_fn {
                    "sigmoid" => {
                        // Single-threaded
                        // self.a[layer +1] = self.z[layer].clone().mapv(|x| sigmoid(x));

                        // Parallel
                        self.a[layer + 1] = self.z[layer].clone();
                        self.a[layer + 1].par_mapv_inplace(|x| sigmoid(x));
                    }
                    "relu" => {
                        // Single-threaded
                        // self.a[layer + 1] = self.z[layer].clone().mapv(|x| relu(x));
                        // Parallel
                        self.a[layer + 1] = self.z[layer].clone();
                        self.a[layer + 1].par_mapv_inplace(|x| relu(x));
                    }
                    _ => panic!(format!(
                        "This activation function ({}) is not supported",
                        activation_fn
                    )),
                }
            }
        } else {
            panic!("Forward pass operation has invalid array sizes");
        }
    }

    /// Begins the training process of the network, using the hyperparameters defined during the network's construction and supplied layer configuration data. A progress-bar and error data is printed to the command-line during this process, and can be exited at any time by pressing `q` or `Esc`.
    #[inline]
    pub fn train(&mut self) -> Result<(), Box<dyn Error>> {
        let mut rng = rand::thread_rng();
        let mut num: usize;
        let validation_floor =
            (self.input.nrows() as f32 * (1. - self.validation_pct)).floor() as usize;
        let validation_ceiling = self.input.nrows();
        assert_eq!(validation_ceiling, self.output.nrows());

        println!("- The number of records in the validation set is: {}, or records {}-{} of the input data",validation_ceiling-validation_floor,validation_floor,validation_ceiling);
        // let validation_set_data = self.input.clone().slice(s![validation_floor..self.input.nrows(), ..]).to_owned();
        let validation_set_labels = self
            .clone()
            .output
            .slice(s![validation_floor..validation_ceiling, ..])
            .to_owned();

        let _stdout = stdout();
        enable_raw_mode()?;

        let pb = ProgressBar::new(self.iterations as u64);
        let sty = ProgressStyle::default_bar()
            .template("[{bar:55}] {msg}")
            .progress_chars("=> ");
        pb.set_style(sty.clone());

        println!("- Beginning to train network, can exit by pressing 'q'");
        for iteration in 0..=self.iterations {
            num = rng.gen_range(0, validation_floor - self.batch_size);
            self.forward_pass(num, self.batch_size);
            self.backwards_pass(num, self.batch_size);

            self.forward_pass(validation_floor, validation_ceiling - validation_floor);
            let error = &self.a[self.l - 1] - &validation_set_labels;
            let error = error.sum().abs() / self.batch_size as f32;

            // Increment the progress bar items
            pb.set_message(&format!(
                "{}/{}    Error: {:.3}",
                iteration, self.iterations, error
            ));
            pb.inc(1);

            // If the network is past the number of minimum training iterations, and the error threshold
            // is below the minimum value, stop training
            if iteration > self.min_iterations {
                if error < self.error_threshold {
                    break;
                }
            }

            if poll(Duration::from_millis(0))? {
                let event = read()?;
                if event == Event::Key(KeyCode::Char('q').into())
                    || event == Event::Key(KeyCode::Esc.into())
                {
                    break;
                }
            }
        }
        disable_raw_mode()?;
        pb.finish_at_current_pos();
        Ok(())
    }

    /// Evaluates the network by running a forward pass on a supplied input using the (trained) network weights
    pub fn evaluate(mut self, input: Array2<f32>) -> Array2<f32> {
        self.input = input.clone();
        self.a[0] = input.clone();
        self.forward_pass(0, 10_000);
        softmax(&mut self.a[self.l - 1]);
        self.a.last().unwrap().clone()
    }
}
