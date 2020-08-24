/// A fully-connected layer, which contains information on the activation function to be applied and the size of the resulting array (number of output nodes)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FCLayer {
    pub activation_function: String,
    pub output_size: usize,
}

impl FCLayer {
    /// Creates a new fully-connected layer configuration with an activation function and number of output nodes
    pub fn new(activation_function: &str, output_size: usize) -> Self {
        FCLayer {
            activation_function: activation_function.to_string(),
            output_size: output_size,
        }
    }

    /// Builds the fundamental structure of a fully-connected layer with a single output node and the sigmoid activation function
    pub fn default() -> Self {
        FCLayer {
            activation_function: "sigmoid".to_string(),
            output_size: 1,
        }
    }

    /// Sets the output size (number of nodes/size of array) of a fully-connected layer
    pub fn output_size(mut self, output_size: usize) -> Self {
        self.output_size = output_size;
        self
    }

    /// Sets the activation function for a fully-connected layer
    pub fn activation_function(mut self, activation_function: &str) -> Self {
        self.activation_function = activation_function.to_string();
        self
    }

    /// Builds and returns the fully-connected layer structure with all applied configurations
    pub fn build(self) -> FCLayer {
        FCLayer {
            activation_function: self.activation_function,
            output_size: self.output_size,
        }
    }
}
