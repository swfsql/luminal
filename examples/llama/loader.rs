use half::f16;
use luminal::{op::Function, prelude::*};

/// Load the model in the same way dfdx-llama does
pub struct DfdxDeferredLoader {
    /// The path to the model folder
    path: String,
}

impl DfdxDeferredLoader {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
        }
    }
}

impl Loader for DfdxDeferredLoader {
    fn load<M: SerializeModule>(self, model: &M, graph: &mut Graph) {
        let mut serializer = Serializer::default();
        model.serialize(&mut serializer);

        for (s, n) in serializer.state {
            let n_elements = graph
                .graph
                .edges_directed(n, petgraph::Direction::Outgoing)
                .next()
                .unwrap()
                .weight()
                .1
                .n_elements();
            if let Some(inp_func) = graph
                .graph
                .node_weight_mut(n)
                .unwrap()
                .as_any_mut()
                .downcast_mut::<Function>()
            {
                let path = self.path.clone();
                inp_func.1 = Box::new(move |_| {
                    // Get memmapped tensor
                    let bytes = std::fs::read(format!("{path}/{s}")).unwrap();
                    let data: Vec<f32> = if bytes.len() == n_elements * 2 {
                        // Half-precision
                        bytes
                            .chunks_exact(std::mem::size_of::<f16>())
                            .map(|chunk| unsafe {
                                std::mem::transmute::<[u8; 2], f16>([chunk[0], chunk[1]]).to_f32()
                            })
                            .collect()
                    } else if bytes.len() == n_elements * 4 {
                        // Full precision
                        bytes
                            .chunks_exact(std::mem::size_of::<f32>())
                            .map(|chunk| unsafe {
                                std::mem::transmute::<[u8; 4], f32>([
                                    chunk[0], chunk[1], chunk[2], chunk[3],
                                ])
                            })
                            .collect()
                    } else {
                        panic!(
                            "Expected {} or {} bytes, got {} when loading {}{}",
                            n_elements * 2,
                            n_elements * 4,
                            bytes.len(),
                            path,
                            s
                        )
                    };

                    Tensor {
                        data: Box::new(data),
                    }
                });
            };
        }
    }
}
