use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use std::error::Error;

use crate::layer::Layer;
use crate::prelude::*;

/// A fully-connected layer
#[derive(Debug)]
pub struct FCLayer {
    pub input: Array2<f32>,
    pub w: Array2<f32>,
    pub a: Array2<f32>,
    pub z: Array2<f32>,
    pub delta: Array2<f32>,
    pub act_fn: String,
    pub learnrate: f32,
}

impl Layer for FCLayer {
    fn print(&self) {
        println!(
            "FCLayer {{
            input: {},
            w: {},
            a: {},
            z: {},
            delta: {},
            act_fn: {},
            learnrate: {}
        }}",
            self.input, self.w, self.a, self.z, self.delta, self.act_fn, self.learnrate
        );
    }

    /// Forward pass of the fully-connected layer, taking a two-dimensional array input
    /// and producing the activated output matrix
    #[inline]
    fn forward(&mut self, input: Array2<f32>) -> Result<Array2<f32>, Box<dyn Error>> {
        self.input = input;
        self.z = self.input.dot(&self.w);

        // Single-threaded
        self.a = match self.act_fn.as_str() {
            "sigmoid" => self.z.mapv(sigmoid),
            _ => panic!("{} is not implemented", self.act_fn),
        };

        // Parallel
        /* self.a = self.z.clone();
        match self.act_fn.as_str() {
            "sigmoid" => self.a.par_mapv_inplace(sigmoid),
            _ => panic!("{} is not implemented", self.act_fn),
        };*/

        Ok(self.a.clone())
    }

    /// Backwards pass of the fully-connected layer, passed down from the above layer
    /// and producing a gradient for the layer below it
    #[inline]
    fn backward(&mut self, input_gradient: Array2<f32>) -> Result<Array2<f32>, Box<dyn Error>> {
        match self.act_fn.as_str() {
            "sigmoid" => {
                self.delta = input_gradient * self.z.mapv(sigmoid_prime);
            }
            _ => panic!("{} is not implemented", self.act_fn),
        }

        let dw = self.input.t().dot(&self.delta).mapv(|x| x * self.learnrate);
        self.w = self.w.clone() - dw;
        let output_gradient = self.delta.dot(&self.w.t());

        Ok(output_gradient)
    }
}

impl FCLayer {
    /// Instantiate a new fully-connected layer by providing number of input/outputs
    /// activation function, and learnrate
    pub fn new(
        shape_io: (usize, usize),
        act_fn: String,
        learnrate: f32,
    ) -> Result<Box<Self>, Box<dyn Error>> {
        Ok(Box::new(FCLayer {
            input: Default::default(),
            act_fn,
            w: Array::random((shape_io.0, shape_io.1), Uniform::new(-0.2, 0.2)),
            z: Default::default(),
            a: Default::default(),
            delta: Default::default(),
            learnrate,
        }))
    }
}

#[test]
fn test_fc_forward() {
    let learnrate = 0.001;
    let shape_io = (15, 10);
    let mut fc_layer = FCLayer::new(shape_io, "sigmoid".to_string(), learnrate).unwrap();
    dbg!(&fc_layer.a);
    let input = Array2::random((1, shape_io.0), Uniform::new(-0.5, 0.5));
    let _output = fc_layer.forward(input).unwrap();
    dbg!(&fc_layer.a); // Shows that forward pass creates properly-shaped output
    assert!(fc_layer.a.shape() == &[1, shape_io.1])
}

#[test]
fn test_fc_backward() {
    let learnrate = 0.1;
    let shape_io = (100, 10);
    let mut fc_layer = FCLayer::new(shape_io, "sigmoid".to_string(), learnrate).unwrap();
    let input = Array2::random((1, shape_io.0), Uniform::new(-0.5, 0.5));
    let actual = Array2::random((1, shape_io.1), Uniform::new(-0.5, 0.5));

    let mut error_cache = 10.0;
    for i in 1..10_000 {
        let output = fc_layer.forward(input.clone()).unwrap();
        let error: Array2<f32> = &output - &actual;
        let delta_init = &error * &fc_layer.z.mapv(|x| sigmoid_prime(x)) * learnrate;
        // Every 1000 iterations, let's check to make sure our error is actually decreasing
        if i % 1_000 == 0 {
            let error_sum = error.sum().abs();
            if error_cache < error_sum {
                println!(
                    "Error @ iteration #{}, {} > {} during iteration {}",
                    i,
                    error_sum,
                    error_cache,
                    i + 1000
                );
                panic!("Error is not decreasing");
            } else {
                println!(
                    "Error @ iteration #{}, {} < {} during iteration {}",
                    i,
                    error_sum,
                    error_cache,
                    i + 1000
                );
            }
            error_cache = error_sum;
        }
        // println!("error: {}",error.sum());
        fc_layer.backward(delta_init).unwrap();
    }
}
