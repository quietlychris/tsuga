use ndarray::prelude::*;
use std::error::Error;

use crate::prelude::convolution::*;

pub fn convolution_2d(
    input: Array2<f32>,
    hp: &ConvHyperParam,
) -> Result<Array2<f32>, Box<dyn Error>> {
    let input = pad_2d(input, hp.padding);

    let (i_n, i_m) = (input.nrows(), input.ncols());
    let (k_n, k_m) = (hp.kernel.nrows(), hp.kernel.ncols());
    // These o_n,m terms don't include the padding values, since we're already taking those
    // into account when we calculuate the input (n,m) values here
    // Otherwise, we'd have an extra (2 * padding) term in the numerator
    let o_n = ((i_n - k_n) as f32 / hp.stride.0 as f32).floor() as usize + 1;
    let o_m = ((i_m - k_m) as f32 / hp.stride.1 as f32).floor() as usize + 1;

    let mut output: Array2<f32> = Array2::zeros((o_n, o_m));
    run_convolution_2d(&hp, &input, &mut output);

    Ok(output)
}

fn run_convolution_2d(hp: &ConvHyperParam, input: &Array2<f32>, output: &mut Array2<f32>) {
    let (i_n, i_m) = (input.shape()[0], input.shape()[1]);
    let (k_n, k_m) = (hp.kernel.shape()[0], hp.kernel.shape()[1]);
    let (o_n, o_m) = (output.shape()[0], output.shape()[1]);

    // println!("{:#?}", output);
    for y in 0..o_n {
        for x in 0..o_m {
            let (i_y, i_x) = (y * hp.stride.0, x * hp.stride.1);
            let temp = &input.slice(s![i_y..(i_y + k_n), i_x..(i_x + k_m)]) * &hp.kernel;
            output[[y, x]] = temp.sum();
            // output.slice_mut(s![x..x+k_m, y..y+k_n]).assign(&temp);
        }
    }
}
