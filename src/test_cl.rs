// Test module for the OpenCL code.
// Most of these test specify an Intel integrated GPU as the unit
// on which to run the tests.
// However, if you're running a system with different hardware (AMD or NVIDIA),
// you may need to make some changes. The option for an NVIDA GeForce GPU
// has already been coded in the `linalg_ocl` module, and should be fairly
// easy to modify for anyone running a different kind of system

use crate::prelude::*;
use crate::linalg_ocl::*;

use std::iter::FromIterator;
use std::time::{Duration, Instant};

use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use ocl::{Device, Platform};

#[test]
#[serial]
fn test_dot_product_small() {
    // let a: Array2<f32> = array![[1., 2., 3., 4.], [4., 3., 2., 1.], [1., 2., 2.5, 4.]];
    // let b: Array2<f32> = array![[1., 1.], [1., 1.], [1., 1.],[1.,1.]];
    let a = Array::random((10, 3), Uniform::new(0., 1.));
    let b = Array::random((3, 2), Uniform::new(0., 1.));

    let (n, m, k): (usize, usize, usize) = (a.nrows(), b.nrows(), b.ncols());
    println!("(n,m,k) = ({},{},{})", n, m, k);
    let c_ndarray = a.dot(&b);

    let a = Array::from_iter(a.iter().cloned()).to_vec();
    let b = Array::from_iter(b.iter().cloned()).to_vec();

    let mut ctx: ocl::ProQue = build_ocl_proque("Intel".to_string());

    let c_vec = dot_product(&mut ctx, &a, &b, (n, m, k)).expect("Couldn't multiply a.dot(b)");
    println!("c_vec:\n{:?}", &c_vec);
    println!("c_ndarray:\n{:?}\n\n\n", create_vec(&c_ndarray));
    let c: Array2<f32> = Array::from_shape_vec((n, k), c_vec)
        .expect("Coudn't convert result to properly sized array");

    let epsilon = 1e-6;
    for y in 0..n {
        for x in 0..k {
            println!(
                "{} - {} = {} < {}",
                c[[y, x]],
                c_ndarray[[y, x]],
                c[[y, x]] - c_ndarray[[y, x]],
                epsilon
            );
            if ((c[[y, x]] - c_ndarray[[y, x]]).abs() > epsilon) {
                panic!(format!(
                    "{} - {} = {} > {}",
                    c[[y, x]],
                    c_ndarray[[y, x]],
                    (c[[y, x]] - c_ndarray[[y, x]]).abs(),
                    epsilon
                ));
            }
        }
    }
}

pub fn create_vec(arr: &Array2<f32>) -> Vec<f32> {
    Array::from_iter(arr.iter().cloned()).to_vec()
}

#[test]
#[serial]
fn test_transpose_then_dot() {
    let a = Array::random((3, 4), Uniform::new(0., 1.));
    let delta = Array::random((3, 2), Uniform::new(0., 1.));
    let (n, m, k): (usize, usize, usize) = (a.nrows(), a.ncols(), delta.ncols());
    let c_ndarray = a.t().dot(&delta);
    let mut a_vec = create_vec(&a);
    let d_vec = create_vec(&delta);
    let mut ctx: ocl::ProQue = build_ocl_proque("Intel".to_string());
    a_vec = transpose(&mut ctx, &a_vec, (n, m)).expect("Couldn't transpose a");
    assert!(a_vec == create_vec(&a.t().to_owned()));
    //******************

    // a_vec is a [4x3] of (m,n)
    // d_vec is a [3x2] of (n,k)
    let c_vec =
        dot_product(&mut ctx, &a_vec, &d_vec, (m, n, k)).expect("Couldn't multiply a.dot(b)");
    println!("raw c_vec:\n{:?}", c_vec);
    println!("vec from ndarray ops:\n{:?}", create_vec(&c_ndarray));
    let c: Array2<f32> = Array::from_shape_vec((m, k), c_vec)
        .expect("Coudn't convert result to properly sized array");
    println!("c_ndarray:\n{:#?}\n", c_ndarray);
    println!("c:\n{:#?}\n\n", c);
}


#[test]
#[serial]
fn test_matmul_large() {
    let iterations = 1;
    let a = Array::random((60_000, 784), Uniform::new(0., 1.));
    let b = Array::random((784, 10), Uniform::new(0., 1.));
    let (n,m,k): (usize,usize,usize) = (a.nrows(), a.ncols(), b.ncols());

    let a_start = Instant::now();
    for _ in 0..iterations {
        let c_ndarray = a.dot(&b);
    }
    let a_end = a_start.elapsed().as_millis();
    println!(
        "Time for {} loops on CPU: {}",
        iterations,
        a_end
    );

    let a = create_vec(&a);
    let b = create_vec(&b);

    // let mut ocl_pq: ocl::ProQue = build_ocl_proque("Intel".to_string());
    let mut ocl_pq: ocl::ProQue = build_ocl_proque("GeForce".to_string());
    let b_start = Instant::now();
    for _ in 0..iterations {
        let c = dot_product(&mut ocl_pq, &a, &b,(n,m,k)).expect("Couldn't multiply a.dot(b)");
    }
    let b_end = b_start.elapsed().as_millis();
    println!(
        "Time for {} loops on GPU: {}",
        iterations,
        b_end
    );

    match a_end < b_end {
        true => println!("The CPU computation is {} times quicker",b_end as f32/ a_end as f32),
        false => println!("The GPU computaiton is {} quicker",a_end as f32/b_end as f32),
        _ => panic!("Something's gone wrong...")
    }

}


#[test]
#[serial]
fn test_multiply_by_scalar() {
    let mut ocl_pq: ocl::ProQue = build_ocl_proque("Intel".to_string());
    let mut v = vec![1.; 20];
    let scalar = 0.5;
    let half_v = match multiply_by_scalar(&mut ocl_pq, v.clone(), scalar) {
        Ok(half_v) => half_v,
        Err(err) => panic!("{}", err),
    };
    v = v.iter().map(|x| *x * scalar).collect();
    // println!("v:\n{:?}",v);
    // println!("half_v:\n{:?}",half_v);
    assert_eq!(half_v, v);
}

#[test]
#[serial]
fn test_hadamard() {
    let a: Array2<f32> = array![[1., 2., 3.], [4., 5., 6.]];
    let b: Array2<f32> = array![[2., 1., 0.666], [0.5, 1., 0.333]];
    let (n, m): (usize, usize) = (a.nrows(), a.ncols());
    assert!(a.dim() == b.dim());

    let c_ndarray = &a * &b;

    let a = Array::from_iter(a.iter().cloned()).to_vec();
    let b = Array::from_iter(b.iter().cloned()).to_vec();

    let mut ocl_pq: ocl::ProQue = build_ocl_proque("Intel".to_string());
    let c_vec = hadamard(&mut ocl_pq, &a, &b, (n, m)).expect("Couldn't multiply a*b");
    let c: Array2<f32> = Array::from_shape_vec((n, m), c_vec)
        .expect("Coudn't convert result to properly sized array");
    assert_eq!(c, c_ndarray);
}

#[test]
#[serial]
fn test_transpose() {
    //let a: Array2<f32> = array![[1.,2.,3.],[4.,5.,6.]];
    let a = Array::random((8, 10), Uniform::new(0., 1.));
    let (n, m): (usize, usize) = (a.nrows(), a.ncols());

    let c_ndarray = a.t();
    println!("c_ndarray:\n{:?}", c_ndarray);

    let a = Array::from_iter(a.iter().cloned()).to_vec();
    let mut ocl_pq: ocl::ProQue = build_ocl_proque("Intel".to_string());
    let c_vec = transpose(&mut ocl_pq, &a, (n, m)).expect("Couldn't transpose a");
    let c: Array2<f32> = Array::from_shape_vec((m, n), c_vec)
        .expect("Coudn't convert result to properly sized array");
    assert_eq!(c, c_ndarray);
}

fn sigmoid_op(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
#[serial]
fn test_sigmoid() {
    let a: Array2<f32> = Array::random((8, 10), Uniform::new(0.49, 0.51));
    let (n, m): (usize, usize) = (a.nrows(), a.ncols());

    let c_ndarray = a.mapv(|x| sigmoid_op(x));

    let a = Array::from_iter(a.iter().cloned()).to_vec();
    let mut ocl_pq: ocl::ProQue = build_ocl_proque("Intel".to_string());
    let c_vec = crate::linalg_ocl::sigmoid(&mut ocl_pq, &a, (n, m)).expect("Couldn't run the sigmoid(a) operation");
    let c: Array2<f32> = Array::from_shape_vec((n, m), c_vec)
        .expect("Coudn't convert result to properly sized array");

    let epsilon = 1e-5;
    for y in 0..n {
        for x in 0..m {
            println!(
                "{} - {} = {} < {}",
                c[[y, x]],
                c_ndarray[[y, x]],
                c[[y, x]] - c_ndarray[[y, x]],
                epsilon
            );
            assert!((c[[y, x]] - c_ndarray[[y, x]]).abs() < epsilon)
        }
    }
}

#[test]
#[serial]
fn test_add() {
    let a: Array2<f32> = array![[1., 2., 3.], [4., 5., 6.]];
    let b: Array2<f32> = array![[2., 1., 0.666], [0.5, 1., 0.333]];
    let (n, m): (usize, usize) = (a.nrows(), a.ncols());
    assert!(a.dim() == b.dim());

    let c_ndarray = &a + &b;

    let a = Array::from_iter(a.iter().cloned()).to_vec();
    let b = Array::from_iter(b.iter().cloned()).to_vec();

    let mut ocl_pq: ocl::ProQue = build_ocl_proque("Intel".to_string());
    let c_vec = add(&mut ocl_pq, &a, &b, (n, m)).expect("Couldn't multiply a*b");
    let c: Array2<f32> = Array::from_shape_vec((n, m), c_vec)
        .expect("Coudn't convert result to properly sized array");
    assert_eq!(c, c_ndarray);
}

#[test]
#[serial]
fn test_subtract() {
    let a: Array2<f32> = array![[1., 2., 3.], [4., 5., 6.]];
    let b: Array2<f32> = array![[2., 1., 0.666], [0.5, 1., 0.333]];
    let (n, m): (usize, usize) = (a.nrows(), a.ncols());
    assert!(a.dim() == b.dim());

    let c_ndarray = &a - &b;

    let a = Array::from_iter(a.iter().cloned()).to_vec();
    let b = Array::from_iter(b.iter().cloned()).to_vec();

    let mut ocl_pq: ocl::ProQue = build_ocl_proque("Intel".to_string());
    let c_vec = subtract(&mut ocl_pq, &a, &b, (n, m)).expect("Couldn't multiply a*b");
    let c: Array2<f32> = Array::from_shape_vec((n, m), c_vec)
        .expect("Coudn't convert result to properly sized array");
    assert_eq!(c, c_ndarray);
}


#[test]
#[serial]
fn small_full_connected_network_w_opencl() {
    // Remember, the `n` value of both the input and output sets must be the same
    //let input = Array::random((20, 10), Uniform::new(0., 1.));
    //let output = Array::random((20, 2), Uniform::new(0., 1.));

    let input: Array2<f32> = array![[1., 2., 3., 4.], [4., 3., 2., 1.], [1., 2., 2.5, 4.]];
    let output: Array2<f32> = array![[1.0, 0.0], [0., 1.0], [1.0, 0.0]];

    let mut layers_cfg: Vec<FCLayer> = Vec::new();
    //let relu_layer_0 = FCLayer::new("relu", 5);
    //layers_cfg.push(relu_layer_0);
    let sigmoid_layer_1 = FCLayer::new("sigmoid", 6);
    layers_cfg.push(sigmoid_layer_1);

    let mut network = FullyConnectedNetwork::default(input.clone(), output.clone())
        //.add_layers(layers_cfg)
        .iterations(100)
        .build();

    let model = network.train_on_gpu("Intel");
    println!("Trained network is:\n{:#?}", network);

    let train_network_repoduced_result = model.clone().evaluate(input);

     println!("Ideal training output:\n{:#?}",output);
    println!("Training set fit:\n{:#?}",network.a[network.l-1]);
    assert_eq!(
        train_network_repoduced_result.mapv(|x| threshold(x, 0.5)),
        network.a[network.l - 1].mapv(|x| threshold(x, 0.5))
    );
    // println!("Reproduced trained network result from model:\n{:#?}",train_network_repoduced_result);

    let test_input: Array2<f32> = array![[4., 3., 3., 1.], [1., 2., 1., 4.]];
    let test_output: Array2<f32> = array![[0.0, 1.0], [1.0, 0.0]];
    let test_result = model.evaluate(test_input);

    println!("Test result:\n{:#?}",test_result);
    println!("Ideal test output:\n{:#?}",test_output);
}