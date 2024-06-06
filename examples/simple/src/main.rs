use luminal::prelude::*;
use luminal_nn::Linear;

fn main() {
    // Create a new graph
    let cx = Graph::new();
    // Randomly initialize a linear layer with an input size of 4 and an output size of 5
    let model = Linear::<4, 5>::initialize(&cx);
    // Make an input tensor
    let a = cx.tensor::<R1<4>>().set(vec![1., 2., 3., 4.]);
    // Feed tensor through model
    let b = model.forward(a).retrieve();

    // Display the graph to see the ops
    cx.display();
    // Execute the graph
    cx.execute_debug();
    // Print the results
    println!("B: {:?}", b.data());
}
