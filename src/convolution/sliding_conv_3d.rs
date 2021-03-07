use ndarray::prelude::*;
use std::error::Error;

use crate::prelude::convolution::*;

pub fn convolution_3d(
    input: Array3<f32>,
    hp: &ConvHyperParam,
) -> Result<Array3<f32>, Box<dyn Error>> {
    let input = pad_3d(input, hp.padding);

    let (stride_n, stride_m) = (hp.stride.0, hp.stride.1);
    let i_dims = input.dim();
    let (i_n, i_m) = (i_dims.1, i_dims.2);
    let (k_n, k_m) = (hp.kernel.nrows(), hp.kernel.ncols());
    // These o_n,m terms don't include the padding values, since we're already taking those
    // into account when we calculuate the input (n,m) values here
    // Otherwise, we'd have an extra (2 * padding) term in the numerator
    let o_n = ((i_n - k_n) as f32 / stride_n as f32).floor() as usize + 1;
    let o_m = ((i_m - k_m) as f32 / stride_m as f32).floor() as usize + 1;

    let mut output: Array3<f32> = Array3::zeros((3, o_n, o_m));
    run_convolution_3d(&hp, &input, &mut output);

    Ok(output)
}

fn run_convolution_3d(hp: &ConvHyperParam, input: &Array3<f32>, output: &mut Array3<f32>) {
    let i_dims = input.dim();
    let i_c = i_dims.0; // number of channels (colors)
    let (i_n, i_m) = (i_dims.1, i_dims.2);
    let (k_n, k_m) = (hp.kernel.shape()[0], hp.kernel.shape()[1]);
    let o_dims = output.dim();
    let (o_n, o_m) = (o_dims.1, o_dims.2);

    for c in 0..i_c {
        for y in 0..o_n {
            for x in 0..o_m {
                let (i_y, i_x) = (y * hp.stride.0, x * hp.stride.1);
                let temp = &input.slice(s![c, i_y..(i_y + k_n), i_x..(i_x + k_m)]) * &hp.kernel;
                output[[c, y, x]] = temp.sum();
            }
        }
    }
}
