#![allow(unused_imports)]

use crate::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::iter::FromIterator;

#[test]
fn check_return_conv_input_2d_2d_shapes() {

    let input = Array::random((28, 28), Uniform::new(0., 1.));

    for padding in {0..3} {
        let hp = ConvHyperParam::new(padding,(1,1), Array::random((3, 3), Uniform::new(0.0, 0.5)) );
        let output = return_conv_input_2d(&input, &hp).unwrap();
        match padding {
            0 => assert_eq!(output.shape(), &[9,676]),
            1 => assert_eq!(output.shape(), &[9,784]),
            2 => assert_eq!(output.shape(), &[9,900]),
            _ => panic!("unexpected padding value")
        }
        
    }
}

#[test]
#[should_panic]
fn check_1d_convolution() {

    // This should panic because the height of the array is less than that of the kernel
    let input = Array::random((1,784), Uniform::new(0., 1.));
    let hp = ConvHyperParam::new(0, (1,1), Array::random((3, 3), Uniform::new(0.0, 0.5)) );
    let output = return_conv_input_2d(&input, &hp).unwrap();
 

}


#[test]
#[serial]
fn sliding_2d_k22s11p0() {
    #[rustfmt::skip]
    let input = array![
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 2.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 3.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 4.0, 0.0]
    ];

    #[rustfmt::skip]
    let kernel = array![
        [1.0, 2.0],
        [3.0, 4.0]
    ];

    let hp = ConvHyperParam::default(kernel).stride((1, 1)).padding(0).build();
    let sliding_output = convolution_2d(input, &hp).unwrap();
    println!("sliding output:\n{:?}", sliding_output);

    #[rustfmt::skip]
    let ideal = array![
        [14.0, 13.0, 10.0, 4.0],
        [12.0, 19.0, 16.0, 4.0],
        [10.0, 14.0, 24.0, 13.0],
    ];
    let eps = 1e-5;
    let diff = (ideal - sliding_output).sum().abs();
    println!("diff: {}", &diff);
    assert!(diff < eps);
}

#[test]
#[serial]
fn sliding_2d_k33s11p0() {
    #[rustfmt::skip]
    let input = array![
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 2.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 3.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 4.0, 0.0]
    ];

    #[rustfmt::skip]
    let kernel = array![
        [1.0, 1.0, 1.0], 
        [0.0, 0.0, 0.0], 
        [-1.0, -1.0, -1.0]
    ];

    let hp = ConvHyperParam::default(kernel).stride((1, 1)).padding(0).build();
    let sliding_output = convolution_2d(input, &hp).unwrap();
    println!("3x3 kernel sliding output:\n{:?}", sliding_output);

    #[rustfmt::skip]
    let ideal = array![
        [-2.0, -2.0, -2.0],
        [1.0, -2.0, -3.0]
    ];
    let eps = 1e-5;
    let diff = (ideal - sliding_output).sum().abs();
    println!("diff: {}", &diff);
    assert!(diff < eps);
}

#[test]
#[serial]
fn small_mm_2d_test() {
    /*
    #[rustfmt::skip]
    let input = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ];
    #[rustfmt::skip]
    let kernel = array![
        [1.0, 2.0],
        [3.0, 4.0]
    ];
    // let kernel = array![[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]];
    */

    let input = Array::random((4, 4), Uniform::new(0., 1.));
    let kernel = Array::random((3, 3), Uniform::new(0., 1.));
    println!("kernel:\n{:#.2?}", &kernel);

    let hp = ConvHyperParam::default(kernel).stride((1, 1)).padding(0).build();

    let sliding_output = convolution_2d(input.clone(), &hp).unwrap();
    println!("output from sliding convolution:\n{:#.2?}", sliding_output);

    let mm_output = mm_convolution_2d(input, &hp).unwrap();
    println!("mm_output:\n{:#.2?}", mm_output);

    let eps = 1e-5;
    let diff = (sliding_output - mm_output).sum();
    println!("diff: {}", &diff);
    assert!(diff.abs() < eps);
}