### Tsuga
#### An early stage, feature-incomplete machine-learning library

**Note: Apologies for the current rough state of the code. It's still under development, so the comments and structure of certain elements are a bit disorganized, particularly around convolution, and only the 012s example is really working at the moment**

Tsuga is an early stage machine learning library in Rust. It uses `ndarray` as the linear algebra backend, and operates primarily on two-dimensional `f64` arrays (`Array2<f64>` types). At the moment, it's primary function has been for testing out various ideas for APIs and as an educational exercise for understanding the structure and process of convolutional neural networks and isn't suitable for serious use.

Initial testing on small fully-connected networks, with a three-number subset of the MNIST data-set which has produced over 90% accuracy on the test set. This example can be run using:
```
$ cargo run --release --example zeros_ones_and_twos
```

At this point, most of the project's focus is on the image-processing domain (particuarly well-suited to 2D arrays), although the tools  and layout should generally applicable to higher/lower-dimensional datasets as well.

Tsuga currently is primarily broken into five stages:

1. Data import and pre-processing to `Array2<f64>`
    - This is done on an individual basis, since the location and format of the data is typically project-specific
2. Data convolution
    - Input arrays use convolutional layers (see features below) to reduce the number of features of each training record
    - Takes one input and uses a user-defined set of convolutional layers to produce one output
3. Each record is flattened at added to a input 2D array of `n x m` dimensions, with a corresponding output array of dimension `n x o` where `o` represents the number of categories that the input record can be mapped to (one-hot encoding).
    - This step eliminates the spatial context of the data
4. The input array is iterated through a set of fully-connected layers, where the user can define the output size of each layer and the activation function, where gradient descent minimizes the difference between the ideal output array and the final output of the fully-connected network. After training is completed, the network returns a model of it's layers, with trained weights and activation functions.
5. The model is used to evaluate an input with the same `m` dimensionality. This can be a test set similar to the training set, to validate the model's ability to classify data other than the training set.

Tsuga's training model is based on fairly simple, easy to understand linear algebra that can be inspected during any stage of the training. A network of L layers has four primary sets of matrices: `a`, `w`, `z`, and `delta`. Bias vectors are not currently used in the calculation.

```
# Forward pass, "·" is the matrix dot product and f(), f'() is an activation function and the derivative of that function respectively
input -> a[0]
z[1] = a[0]·W[0]
a[1] = f(z[1])
   ...
a[L] = f(z[L])

# Backwards pass, ^T indicates a matrix transpose, "*" is element-wise multiplication or the Hadamard product
delta[L] = (a[L] - output) * f'(z[L]) * learnrate
w[L-1] = w[L-1] - a[L-1]^T · delta[L]
   ...
w[0] = w[0] - a[0]^T · delta[1]
```

#### Features

- [X] Fully connected network
    - [x] Preliminary API
    - [x] Fully connected layers
    - [x] Forward passes and error backpropogation with gradient descent
    - [ ] Stochastic gradient descent (not fully implemented)
    - [x] Sigmoid logistic and ReLU activation functions
    - [x] Preliminary API
    - [x] In-program saving and export of network model
    - [ ] External saving and import of model as binary + human-readable
    - [ ] Bias vectors used in the forward/backward passes
    - [ ] Complete coverage of matrix compatibility at compile-time with testing
    - [ ] Error handling with model state save if panic occurs

- [ ] Convolutional networks
    - [x] Preliminary API/convenience methods for chaining simple convolutions
    - [x] Customized 2D kernels
    - [ ] Arbitrary values of stride and padding
    - [ ] Max pooling and ReLU on convolutional layers
    - [ ] Inspection of intermediate network convolution results
    - [ ] Direct integration of convolutional and fully-connected layers
        - This will probably be done by combining the different layer types into a single enum with multiple variants
    - [ ] 3D -> 2D image convolution
    - [ ] Write example

- [ ] Performance options
    - [ ] Default to parallel methods for linear algebra
    - [ ] GPU compute options (arrayfire-rust bindings?)
    - [ ] Minimize number of array reallocations

#### Fully-connected network example
Tsuga currently uses the [Builder](https://xaeroxe.github.io/init-struct-pattern/) pattern for constructing fully-connected networks. Since these are complex compound structure, the helps to make the layout of the network explicit and modular.

```rust
use tsuga::prelude::*;

fn main() {

    let input: Array2<f64> = array![
        [1., 2., 3., 4.],
        [4., 3., 2., 1.],
        [1., 2., 2.5, 4.]
    ];
    let output: Array2<f64> = array![
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0]
    ];

    let mut layers_cfg: Vec<FCLayer> = Vec::new();
    let relu_layer_0 = FCLayer::new("relu", 2);
    layers_cfg.push(relu_layer_0);
    let sigmoid_layer_1 = FCLayer::new("sigmoid", 2);
    layers_cfg.push(sigmoid_layer_1);

    let mut network = FullyConnectedNetwork::default(input.clone(), output.clone())
        .add_layers(layers_cfg)
        .iterations(10000)
        .build();

    let model = network.train();
    // let model = network.sgd_train(10); // Batch SGD implementation isn't really working at the moment
    println!("Trained network is:\n{:#?}", network);

    let test_input: Array2<f64> = array![
        [4., 3., 3., 1.],
        [1., 2., 1., 4.]
    ];
    let test_output: Array2<f64> = array![
        [0.0, 1.0],
        [1.0, 0.0]
    ];
    let test_result = model.evaluate(test_input);
    println!("Test result:\n{:#?}",test_result);
    println!("Ideal test output:\n{:#?}",test_output);
}

```
