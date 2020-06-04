// Test module for the OpenCL code.
// Most of these test specify an Intel integrated GPU as the unit
// on which to run the tests.
// However, if you're running a system with different hardware (AMD or NVIDIA),
// you may need to make some changes. The option for an NVIDA GeForce GPU
// has already been coded in the `linalg_ocl` module, and should be fairly
// easy to modify for anyone running a different kind of system

use crate::linalg_ocl::*;

use std::iter::FromIterator;
use std::time::{Duration, Instant};

use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use ocl::{Device, Platform};

#[test]
#[serial]
fn test_matmul_small() {
    let a: Array2<f32> = array![[1., 2., 3.], [4., 5., 6.]];
    let b: Array2<f32> = array![[1., 1.], [1., 1.], [1., 1.]];
    let (n, m, k): (usize, usize, usize) = (a.nrows(), a.ncols(), b.ncols());
    let c_ndarray = a.dot(&b);

    let a = Array::from_iter(a.iter().cloned()).to_vec();
    let b = Array::from_iter(b.iter().cloned()).to_vec();

    let mut ocl_pq: ocl::ProQue = build_ocl_proque("Intel".to_string());
    let c_vec = matmul(&mut ocl_pq, &a, &b, (n, m, k)).expect("Couldn't multiply a.dot(b)");
    let c: Array2<f32> = Array::from_shape_vec((n, k), c_vec)
        .expect("Coudn't convert result to properly sized array");
    assert_eq!(c, c_ndarray);
}

/*
#[test]
#[serial]
#[ignore]
fn test_matmul_large() {
    let iterations = 1;
    let a = Array::random((60_000, 784), Uniform::new(0., 1.));
    let b = Array::random((784, 10), Uniform::new(0., 1.));

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


    let mut ocl_pq: ocl::ProQue = build_ocl_proque("Intel".to_string());
    let b_start = Instant::now();
    for _ in 0..iterations {
        let (n,m,k): (usize,usize,usize) = (a.nrows(), a.ncols(), b.ncols());
        let c = matmul(&mut ocl_pq, &a, &b,(n,m,k)).expect("Couldn't multiply a.dot(b)");
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
*/

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
    let c_vec = sigmoid(&mut ocl_pq, &a, (n, m)).expect("Couldn't run the sigmoid(a) operation");
    let c: Array2<f32> = Array::from_shape_vec((n, m), c_vec)
        .expect("Coudn't convert result to properly sized array");

    let epsilon = 1e-5;

    let epsilon = 1e-3;
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
