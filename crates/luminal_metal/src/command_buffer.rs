use std::{any::Any, cell::UnsafeCell, fmt::Debug, ops::Deref, sync::Arc};

use itertools::Itertools;
use metal_rs::{Buffer, CommandBuffer, CommandQueue, Device};
use petgraph::{
    stable_graph::NodeIndex,
    visit::EdgeRef,
    Direction::{self},
};
use rustc_hash::{FxHashMap, FxHashSet};

use luminal::{
    op::{InputTensor, Operator},
    prelude::*,
};

use crate::{MetalBuffer, MetalKernel, MetalKernelWrapper};

use super::get_buffer_from_tensor;

#[derive(Default, Debug)]
pub struct CommandBufferCompiler;

impl Compiler for CommandBufferCompiler {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &GraphWrapper, _: T) {
        let is_metal: FxHashSet<NodeIndex> = graph
            .graph
            .node_indices()
            .collect::<Vec<_>>()
            .into_iter()
            .filter(|i| {
                graph
                    .graph
                    .node_weight_mut(*i)
                    .unwrap()
                    .custom("metal", Box::new(()))
                    .is_some()
            })
            .collect();
        // Do forward pass
        let mut forward_map: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        for node in graph
            .graph
            .node_indices()
            .filter(|n| graph.graph.edges_directed(*n, Direction::Incoming).count() == 0)
            .sorted()
        {
            let mut stack = vec![node];
            while let Some(node) = stack.pop() {
                // Get rank as max of predecessors
                let rank = graph
                    .graph
                    .neighbors_directed(node, Direction::Incoming)
                    .filter_map(|i| forward_map.get(&i).map(|r| (i, *r)))
                    .map(|(node_index, rank)| {
                        if is_metal.contains(&node) != is_metal.contains(&node_index) {
                            rank + 1
                        } else {
                            rank
                        }
                    })
                    .max()
                    .unwrap_or_default();
                // Max it with the current entry in the map or insert
                if let Some(entry) = forward_map.get_mut(&node) {
                    if rank > *entry {
                        *entry = rank;
                        stack.extend(graph.graph.neighbors_directed(node, Direction::Outgoing));
                    }
                } else {
                    forward_map.insert(node, rank);
                    stack.extend(graph.graph.neighbors_directed(node, Direction::Outgoing));
                }
            }
        }

        // Do backward pass
        let mut backward_map: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        for node in graph
            .graph
            .node_indices()
            .filter(|n| graph.graph.edges_directed(*n, Direction::Outgoing).count() == 0)
            .sorted()
        {
            let mut stack = vec![node];
            while let Some(node) = stack.pop() {
                // Get rank as max of successors
                let rank = graph
                    .graph
                    .neighbors_directed(node, Direction::Outgoing)
                    .filter_map(|i| backward_map.get(&i).map(|r| (i, *r)))
                    .map(|(node_index, rank)| {
                        if is_metal.contains(&node) != is_metal.contains(&node_index) {
                            rank + 1
                        } else {
                            rank
                        }
                    })
                    .max()
                    .unwrap_or_default();
                // Max it with the current entry in the map or insert
                if let Some(entry) = backward_map.get_mut(&node) {
                    if rank > *entry {
                        *entry = rank;
                        stack.extend(graph.graph.neighbors_directed(node, Direction::Incoming));
                    }
                } else {
                    backward_map.insert(node, rank);
                    stack.extend(graph.graph.neighbors_directed(node, Direction::Incoming));
                }
            }
        }
        // Get sets (Rank -> # of nodes with that rank)
        let forward_sets = forward_map
            .iter()
            .sorted_by_key(|(_, v)| **v)
            .group_by(|(_, v)| **v)
            .into_iter()
            .map(|(k, g)| (k, g.count()))
            .collect::<FxHashMap<_, _>>();
        let backward_sets = backward_map
            .iter()
            .sorted_by_key(|(_, v)| **v)
            .group_by(|(_, v)| **v)
            .into_iter()
            .map(|(k, g)| (k, g.count()))
            .collect::<FxHashMap<_, _>>();

        // Assign nodes to sets
        let mut node_sets: FxHashMap<(bool, usize), FxHashSet<NodeIndex>> = FxHashMap::default();
        for node in graph.graph.node_indices().filter(|i| is_metal.contains(i)) {
            let forward_bigger =
                forward_sets[&forward_map[&node]] >= backward_sets[&backward_map[&node]];
            node_sets
                .entry((
                    forward_bigger,
                    if forward_bigger {
                        forward_map[&node]
                    } else {
                        backward_map[&node]
                    },
                ))
                .and_modify(|set| {
                    set.insert(node);
                })
                .or_insert({
                    let mut set = FxHashSet::default();
                    set.insert(node);
                    set
                });
        }
        // Add sets to graph
        let dev = Device::system_default().unwrap();
        let mut queue = dev.new_command_queue();
        let mut num_buffers_on_queue = 0;
        for set in node_sets.values() {
            if num_buffers_on_queue >= 63 {
                num_buffers_on_queue = 0;
                queue = dev.new_command_queue();
            } else {
                num_buffers_on_queue += 1;
            }
            #[allow(clippy::arc_with_non_send_sync)]
            let buffer = Arc::new(UnsafeCell::new(queue.new_command_buffer().to_owned()));
            let exec = graph
                .add_op(ExecuteMetalKernels {
                    queue: queue.clone(),
                    buffer: buffer.clone(),
                })
                .finish();
            for node in set {
                // Create schedule dependency
                graph.add_schedule_dependency(*node, exec);
                // Wrap node in MetalKernelOperation
                let wrapper = graph
                    .graph
                    .node_weight_mut(*node)
                    .unwrap()
                    .custom("metal", Box::new(()))
                    .unwrap()
                    .downcast::<MetalKernelWrapper>()
                    .unwrap();
                *graph.graph.node_weight_mut(*node).unwrap() = Box::new(CommandBufferWrapper {
                    wrapper,
                    buffer: buffer.clone(),
                    dyn_map: &graph.dyn_map,
                });
                // Create schedule dependencies from exec to consumers
                for outside_node in graph
                    .graph
                    .edges_directed(*node, Direction::Outgoing)
                    .filter(|e| !e.weight().is_schedule())
                    .map(|e| e.target())
                    .filter(|n| !set.contains(n))
                    .collect::<Vec<_>>()
                {
                    graph.add_schedule_dependency(exec, outside_node);
                }
            }
        }
    }
}

struct ExecuteMetalKernels {
    queue: CommandQueue,
    buffer: Arc<UnsafeCell<CommandBuffer>>,
}
impl Debug for ExecuteMetalKernels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ExecuteMetalKernels")
    }
}

impl Operator for ExecuteMetalKernels {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let buffer = unsafe { &mut *self.buffer.get() };
        buffer.commit();
        buffer.wait_until_completed();
        *buffer = self.queue.new_command_buffer().to_owned();
        vec![]
    }
}

#[derive(Clone)]
struct CommandBufferWrapper {
    wrapper: Box<MetalKernelWrapper>,
    buffer: Arc<UnsafeCell<CommandBuffer>>,
    dyn_map: *const FxHashMap<char, usize>,
}

impl std::fmt::Debug for CommandBufferWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalKernel({:?})", self.wrapper.0)
    }
}

impl MetalKernel for CommandBufferWrapper {
    fn intermediate_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        self.wrapper.intermediate_buffer_sizes(input_shapes)
    }
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<BigExpression> {
        self.wrapper.output_buffer_sizes(input_shapes)
    }
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        _: &metal_rs::CommandBufferRef,
        intermediate_buffers: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        self.wrapper.metal_forward(
            inputs,
            unsafe { &*self.buffer.get() },
            intermediate_buffers,
            output_buffers,
        );
    }
    fn without_command_buffer(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        intermediate_buffers: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        self.metal_forward(
            inputs,
            unsafe { &*self.buffer.get() },
            intermediate_buffers,
            output_buffers,
        )
    }
}

impl Operator for CommandBufferWrapper {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        self.without_storage_buffers(
            &inp.iter()
                .map(|(t, sh)| (get_buffer_from_tensor(t).deref(), *sh))
                .collect::<Vec<_>>(),
            unsafe { &*self.buffer.get() },
            unsafe { self.dyn_map.as_ref().unwrap() },
        )
        .into_iter()
        .map(|b| Tensor::new(MetalBuffer(b)))
        .collect()
    }

    #[allow(clippy::arc_with_non_send_sync)]
    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

#[cfg(test)]
#[test]
fn test_common_buffer() {
    use luminal::{
        prelude::*,
        tests::{assert_close, random_vec},
    };

    use crate::MetalCompiler;
    let mut cx = Graph::new();
    let a = cx.tensor::<R1<5>>().set(random_vec(5)).keep();
    let b = cx.tensor::<R1<5>>().set(random_vec(5)).keep();
    let c = cx.tensor::<R1<5>>().set(random_vec(5)).keep();
    let mut d = ((a + b) * c).retrieve();

    cx.execute();
    let d_unopt = d.data();
    d.drop();

    cx.compile(MetalCompiler::<f16>::default(), &mut d);
    cx.execute();

    assert_close(&d.data(), &d_unopt);
}
