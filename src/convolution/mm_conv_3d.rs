use ndarray::prelude::*;
use std::error::Error;

use crate::prelude::convolution::*;

pub fn native_mm_convolution_3d(
    input: Array3<f32>,
    hp: &ConvHyperParam,
) -> Result<Array3<f32>, Box<dyn Error>> {
    let input = pad_3d(input, hp.padding);

    let channels = input.dim().0;
    let (i_n, i_m) = (input.dim().1, input.dim().2);
    let (k_n, k_m) = (hp.kernel.nrows(), hp.kernel.ncols());
    let o_n = ((i_n - k_n) as f32 / hp.stride.0 as f32).floor() as usize + 1;
    let o_m = ((i_m - k_m) as f32 / hp.stride.1 as f32).floor() as usize + 1;

    let flat_kernel = return_flat_kernel_2d(hp)?;
    let altered_input = return_conv_input_3d(&input, &hp)?;

    // dbg!(altered_input.shape());
    let output = flat_kernel
        .dot(&altered_input)
        .into_shape((channels, o_n, o_m))?;
    println!("output.shape() = {:?}", output.shape());

    Ok(output)
}

pub fn return_conv_input_3d(
    input: &Array3<f32>,
    hp: &ConvHyperParam,
) -> Result<Array2<f32>, Box<dyn Error>> {
    let (i_n, i_m) = (input.dim().1, input.dim().2);
    // println!("(i_n, i_m) = ({}, {}) = {} total elements",i_n, i_m, i_n * i_m);
    let (k_n, k_m) = (hp.kernel.nrows(), hp.kernel.ncols());

    let o_n = ((i_n - k_n) as f32 / hp.stride.0 as f32).floor() as usize + 1;
    let o_m = ((i_m - k_m) as f32 / hp.stride.1 as f32).floor() as usize + 1;
    // println!("(o_n, o_m) = ({}, {}) = {} total elements",o_n, o_m, o_n * o_m);

    let channels = 3;

    let mut altered_input: Array2<f32> = Array2::zeros((k_n * k_m, o_m * o_n * channels));
    // dbg!(altered_input.shape());

    let n = o_n * o_m;
    for c in 0..channels {
        for y in 0..o_n {
            for x in 0..o_m {
                let (i_y, i_x) = (y * hp.stride.0, x * hp.stride.1);
                let temp = input
                    .slice(s![c, i_y..(i_y + k_n), i_x..(i_x + k_m)])
                    .to_owned()
                    .into_shape((k_n * k_m, 1))?;
                altered_input
                    .slice_mut(s![0..(k_n * k_m), c * n + ((y * o_m) + x)])
                    .assign(&temp.slice(s![.., 0]));
            }
        }
    }

    Ok(altered_input)
}

/// This the three-dimensional variation
fn return_flat_kernel_3d(hp: &ConvHyperParam) -> Result<Array2<f32>, Box<dyn Error>> {
    // Note: this is the three-dimensional

    let (k_n, k_m) = (hp.kernel.nrows(), hp.kernel.ncols());
    let flat_kernel_base = hp.kernel.clone().into_shape((1, k_n * k_m))?;

    let mut flat_kernel = Array2::zeros((3, k_n * k_m));
    for i in 0..3 {
        flat_kernel
            .slice_mut(s![i, 0..(k_n * k_m)])
            .assign(&flat_kernel_base.slice(s![.., 0]));
    }

    Ok(flat_kernel)
}

/// 3d convolution by stacking layers of 2d convolutions
pub fn mm_convolution_3d(
    input: Array3<f32>,
    hp: &ConvHyperParam,
) -> Result<Array3<f32>, Box<dyn Error>> {
    let (i_n, i_m) = (input.dim().1, input.dim().2);
    println!(
        "the input is of shape {:?}, leading to an (i_n, i_m) of {:?}",
        input.shape(),
        (i_n, i_m)
    );

    let (k_n, k_m) = (hp.kernel.nrows(), hp.kernel.ncols());

    let o_n = ((i_n - k_n) as f32 / hp.stride.0 as f32).floor() as usize + 1 + hp.padding;
    let o_m = ((i_m - k_m) as f32 / hp.stride.1 as f32).floor() as usize + 1 + hp.padding;

    let channels = 3;

    // let mut temp_channel_out = Array2::zeros((o_n, o_m));
    let mut output = Array3::zeros((channels, o_n, o_m));
    // println!()
    for c in 0..channels {
        let channel_input = input
            .slice(s![c, .., ..])
            .to_owned()
            .into_shape((i_n, i_m))?;
        println!("the channel input shape is: {:?}", channel_input.shape());
        let channel_output = mm_convolution_2d(channel_input, hp)?;
        println!("the channel output shape is: {:?}", channel_output.shape());
        println!("which we want to put into a shape of: {:?}", output.shape());
        output
            .slice_mut(s![c, .., ..])
            .assign(&channel_output.slice(s![.., ..]));
    }
    Ok(output)
}

// Expects a (3,n,m) dimensioned input
pub fn pad_3d(input: Array3<f32>, padding: usize) -> Array3<f32> {
    match padding {
        0 => input,
        _ => {
            let dims = input.dim();
            println!("input dimensions are: {:?}", dims);
            let (n, m) = (dims.1, dims.2);
            println!("with an (n,m) of ({},{})", n, m);

            let mut out: Array3<f32> = Array3::zeros((3, n + (padding * 2), m + (padding * 2)));
            // Can this be done in parallel using iterators + Rayon?
            for i in 0..3 {
                out.slice_mut(s![i, padding..n + padding, padding..m + padding])
                    .assign(&input.slice(s![i, .., ..]));
            }

            out
        }
    }
}

#[test]
fn small_mm_3d_convolution() {
    let input = array![
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
        [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]
    ];

    let kernel_v = array![[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]];
    let hp_vert = ConvHyperParam::default(kernel_v).stride((1, 1)).build();

    let output = mm_convolution_3d(input, &hp_vert).unwrap();

    println!("output: {:?}", output);
}
