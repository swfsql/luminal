#![allow(clippy::needless_range_loop)]

use crate::prelude::*;
use std::{
    cell::RefCell,
    io::Write,
    ops::{Deref, DerefMut},
    rc::Rc,
    time::Duration,
};

use super::compiler_utils::{ToIds, ToIdsMut};
use colored::Colorize;
use itertools::Itertools;
use petgraph::{stable_graph::StableGraph, visit::EdgeRef, Direction};
use rustc_hash::{FxHashMap, FxHashSet};

pub type MainGraph = StableGraph<Box<dyn Operator>, Dependency>;

// pub struct GraphWrapper(pub Rc<RefCell<Graph>>);
// pub type GraphWrapper = Rc<RefCell<Graph>>;
#[derive(Default)]
pub struct GraphWrapper(pub Rc<RefCell<Graph>>);

impl GraphWrapper {
    /// Try to remove the tensor data from the graph
    pub fn get_tensor(&self, id: NodeIndex, ind: u8) -> Option<Tensor> {
        self.borrow_mut().get_tensor(id, ind)
    }

    /// Try to get the tensor data in the graph
    pub fn get_tensor_ref(&self, id: NodeIndex, ind: u8) -> Option<Tensor> {
        self.borrow().get_tensor_ref(id, ind)
    }

    /// Delete the tensor data from the graph
    pub fn drop_tensors<T: ToIds>(&self, tensors: T) {
        self.borrow_mut().drop_tensors(tensors)
    }

    /// Mark tensors to be kept
    pub fn keep_tensors<T: ToIds>(&self, tensors: T) {
        self.borrow_mut().keep_tensors(tensors)
    }

    /// Set a tensor's data
    pub fn set_tensor(&self, id: NodeIndex, ind: u8, tensor: Tensor) {
        self.borrow_mut().set_tensor(id, ind, tensor)
    }

    /// Set a dynamic dimension
    pub fn set_dyn_dim(&self, dimension: char, val: usize) {
        self.borrow_mut().set_dyn_dim(dimension, val)
    }

    /// Create a new tensor with shape S
    pub fn tensor<S: Shape>(&self) -> GraphTensor<S> {
        self.named_tensor("Tensor")
    }

    /// Create a new tensor with shape S and a name. This name will show up on the graph when displayed
    pub fn named_tensor<S: Shape>(&self, name: &str) -> GraphTensor<S> {
        let mut _self = self.borrow_mut();
        let name = name.to_string();
        let id = _self.graph.add_node(Box::new(Function(
            format!("{name} Load"),
            Box::new(move |_| panic!("You must set a value for this tensor! ({name})")),
        )));
        GraphTensor {
            id,
            graph_ref: Rc::downgrade(self.as_ref()),
            shape: S::to_tracker(),
            _phantom: Default::default(),
        }
    }

    /// Compile the graph using the given compiler
    pub fn compile<T: ToIdsMut, C: Compiler>(&self, compiler: C, remap: T) -> C::Output {
        let output = compiler.compile(self, remap);
        self.toposort();
        self.reset();
        output
    }

    /// Refresh the internally sorted graph
    pub(crate) fn toposort(&self) {
        self.borrow_mut().toposort()
    }

    /// Swap the tensors with these ids
    pub fn swap_tensors<A: Shape, B: Shape>(&self, a: GraphTensor<A>, b: GraphTensor<B>) {
        self.borrow_mut().swap_tensors(a, b)
    }

    /// Clear any remaining tensors that may be around from old executions
    pub fn reset(&self) {
        self.borrow_mut().reset()
    }

    /// Execute the graph.
    pub fn execute(&self) {
        self.borrow_mut().execute()
    }

    /// Execute the graph without deleting intermediate tensors
    pub fn execute_no_delete(&self) {
        self.borrow_mut().execute_no_delete()
    }

    /// Execute the graph with debug prints
    pub fn execute_debug(&self) {
        self.borrow_mut().execute_debug()
    }
}

impl AsRef<Rc<RefCell<Graph>>> for GraphWrapper {
    fn as_ref(&self) -> &Rc<RefCell<Graph>> {
        &self.0
    }
}

impl AsMut<Rc<RefCell<Graph>>> for GraphWrapper {
    fn as_mut(&mut self) -> &mut Rc<RefCell<Graph>> {
        &mut self.0
    }
}

/// A Luminal compute graph.
///
/// All computation is represented as a directed acyclic graph.
/// All data is stored inside this object as well.
#[derive(Debug, Default)]
pub struct Graph {
    /// The store of tensors in the graph. Indexed by node index and output index.
    pub tensors: FxHashMap<(NodeIndex, u8), Tensor>,
    /// A map of dynamic dimensions to concrete dimension sizes
    pub dyn_map: Rc<RefCell<FxHashMap<char, usize>>>,
    /// Edge weights: (Input index, Output index, Input shape)
    pub graph: MainGraph,
    /// Tensors marked in this set will not get deleted when the graph is ran
    pub no_delete: FxHashSet<NodeIndex>,
    /// Tensors marked in this set need to be retrieved later (mostly for optimizers to insert copy back calls, the graph itself doesn't treat these differently)
    pub to_retrieve: FxHashMap<NodeIndex, (u8, ShapeTracker)>,
    /// A list of current node to run, source nodes, and view nodes to delete after execution.
    #[allow(clippy::type_complexity)]
    pub(crate) linearized_graph: Option<Vec<(NodeIndex, Vec<(NodeIndex, u8, ShapeTracker)>)>>,
    /// Cached consumers (for execution only)
    consumers_map: Option<FxHashMap<(NodeIndex, u8), usize>>,
}

/// A dependency between two nodes
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum Dependency {
    /// A data dependency (transferring a tensor from one node to the next)
    Data {
        input_order: u8,
        output_order: u8,
        shape: ShapeTracker,
    },
    /// Explicit dependency for ordering. No tensors are transferred through this dependency
    Schedule,
}

impl Dependency {
    /// Try to extract dependency data
    pub fn as_data(self) -> Option<(u8, u8, ShapeTracker)> {
        if let Self::Data {
            input_order,
            output_order,
            shape,
        } = self
        {
            Some((input_order, output_order, shape))
        } else {
            None
        }
    }

    /// Is this a schedule dependency?
    pub fn is_schedule(&self) -> bool {
        matches!(self, Self::Schedule)
    }
}

impl Graph {
    /// Create a new graph
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> GraphWrapper {
        GraphWrapper(Rc::new(RefCell::new(Graph::default())))
    }

    /// Try to remove the tensor data from the graph
    pub fn get_tensor(&mut self, id: NodeIndex, ind: u8) -> Option<Tensor> {
        self.tensors.remove(&(id, ind))
    }

    /// Try to get the tensor data in the graph
    pub fn get_tensor_ref(&self, id: NodeIndex, ind: u8) -> Option<Tensor> {
        self.tensors.get(&(id, ind)).cloned()
    }

    /// Delete the tensor data from the graph
    pub fn drop_tensors<T: ToIds>(&mut self, tensors: T) {
        for id in tensors.to_ids() {
            self.tensors.remove(&(id, 0));
        }
    }

    /// Mark tensors to be kept
    pub fn keep_tensors<T: ToIds>(&mut self, tensors: T) {
        for id in tensors.to_ids() {
            self.no_delete.insert(id);
        }
    }

    /// Set a tensor's data
    pub fn set_tensor(&mut self, id: NodeIndex, ind: u8, tensor: Tensor) {
        self.tensors.insert((id, ind), tensor);
    }

    /// Set a dynamic dimension
    pub fn set_dyn_dim(&mut self, dimension: char, val: usize) {
        self.dyn_map.borrow_mut().insert(dimension, val);
    }

    /// Refresh the internally sorted graph
    pub(crate) fn toposort(&mut self) {
        self.linearized_graph = Some(
            petgraph::algo::toposort(&self.graph, None)
                .unwrap()
                .into_iter()
                .map(|node| (node, self.get_sources(node)))
                .collect(),
        );

        // Refresh the internal remaining consumers map
        self.consumers_map = Some(
            self.graph
                .node_indices()
                .flat_map(|i| {
                    self.graph
                        .edges_directed(i, Direction::Outgoing)
                        .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i)))
                        .group_by(|(_, (_, i, _))| *i)
                        .into_iter()
                        .map(|(ind, g)| ((i, ind), g.count()))
                        .collect::<Vec<_>>()
                })
                .collect(),
        );
    }

    /// Swap the tensors with these ids
    pub fn swap_tensors<A: Shape, B: Shape>(&mut self, a: GraphTensor<A>, b: GraphTensor<B>) {
        // Swap tensors
        for i in 0.. {
            let a_t = self.tensors.remove(&(a.id, i));
            let b_t = self.tensors.remove(&(b.id, i));
            if a_t.is_none() && b_t.is_none() {
                break;
            }
            if let Some(a_t) = a_t {
                self.tensors.insert((b.id, i), a_t);
            }
            if let Some(b_t) = b_t {
                self.tensors.insert((a.id, i), b_t);
            }
        }
    }

    /// Clear any remaining tensors that may be around from old executions
    pub fn reset(&mut self) {
        self.tensors.retain(|(n, _), _| self.no_delete.contains(n));
    }

    /// Execute the graph.
    pub fn execute(&mut self) {
        // Track the number of views pointing to each tensor so we know when to clear
        if self.linearized_graph.is_none() {
            self.toposort();
        }
        let mut consumers = self.consumers_map.as_ref().unwrap().clone();
        let mut dim_stack = Vec::new();

        for (node, src_ids) in self.linearized_graph.as_ref().unwrap() {
            if self.tensors.contains_key(&(*node, 0)) {
                continue;
            }

            let mut srcs =
                get_source_tensors(&self.no_delete, &mut self.tensors, src_ids, &consumers);

            // Substitute in the dyn dims
            let dyn_map = &self.dyn_map.as_ref().borrow();
            for (_, st) in srcs.iter_mut() {
                st.resolve_global_dyn_dims_stack(dyn_map, &mut dim_stack);
            }

            // Execute
            let tensors = self.graph.node_weight_mut(*node).unwrap().process(srcs);
            for (i, tensor) in tensors.into_iter().enumerate() {
                self.tensors.insert((*node, i as u8), tensor);
            }

            // Bookkeep remaining consumers
            for (id, ind, _) in src_ids {
                *consumers.get_mut(&(*id, *ind)).unwrap() -= 1;
            }
        }
        self.reset();
    }

    /// Execute the graph without deleting intermediate tensors
    pub fn execute_no_delete(&mut self) {
        // Track the number of views pointing to each tensor so we know when to clear;
        if self.linearized_graph.is_none() {
            self.toposort();
        }
        let mut dim_stack = Vec::new();
        for (node, src_ids) in self.linearized_graph.as_ref().unwrap().iter() {
            if self.tensors.contains_key(&(*node, 0)) {
                continue;
            }
            let mut srcs = src_ids
                .iter()
                .map(|(id, ind, st)| {
                    let tensor = self.tensors.get(&(*id, *ind)).unwrap();
                    (InputTensor::new(tensor.clone()), *st)
                })
                .collect_vec();

            // Substitute in the dyn dims
            let dyn_map = &self.dyn_map.as_ref().borrow();
            for (_, st) in srcs.iter_mut() {
                st.resolve_global_dyn_dims_stack(dyn_map, &mut dim_stack);
            }

            // All sources are ready, execute
            let tensors = self.graph.node_weight_mut(*node).unwrap().process(srcs);
            for (i, tensor) in tensors.into_iter().enumerate() {
                self.tensors.insert((*node, i as u8), tensor);
            }
        }
    }

    /// Execute the graph with debug prints
    pub fn execute_debug(&mut self) {
        fn format_duration(duration: &Duration) -> String {
            if duration.as_secs() > 0 {
                format!("{:.2}s", duration.as_secs_f32())
            } else if duration.as_millis() > 0 {
                format!("{}ms", duration.as_millis())
            } else {
                format!("{}µs", duration.as_micros())
            }
        }
        // Track the number of views pointing to each tensor so we know when to clear
        if self.linearized_graph.is_none() {
            self.toposort();
        }
        let mut dim_stack = Vec::new();
        let mut consumers = self.consumers_map.as_ref().unwrap().clone();
        let mut op_times = FxHashMap::default();
        let width = term_size::dimensions().unwrap().0;

        println!(
            "{:->2$} Executing {:->2$}",
            "",
            "",
            (width.saturating_sub(" Executing ".len())) / 2
        );
        let start = std::time::Instant::now();
        for (node, src_ids) in self.linearized_graph.as_ref().unwrap().iter() {
            if self.tensors.contains_key(&(*node, 0)) {
                continue;
            }
            let op_name = format!("{:?} | {}", self.node_weight(*node).unwrap(), node.index());
            print!("{}", op_name.bold().bright_green());

            let mut srcs =
                get_source_tensors(&self.no_delete, &mut self.tensors, src_ids, &consumers);

            // Substitute in the dyn dims
            let dyn_map = &self.dyn_map.as_ref().borrow();
            for (_, st) in srcs.iter_mut() {
                st.resolve_global_dyn_dims_stack(dyn_map, &mut dim_stack);
            }

            // All sources are ready
            let mut shapes_string = srcs
                .iter()
                .map(|(_, s)| {
                    format!(
                        "{:?}",
                        s.shape()
                            .into_iter()
                            .map(|i| i.to_usize().unwrap())
                            .collect::<Vec<_>>()
                    )
                })
                .join(", ");
            if !shapes_string.is_empty() {
                shapes_string = format!(" ({shapes_string})");
            }
            print!("{shapes_string}");
            std::io::stdout().flush().unwrap();
            // Execute
            let now = std::time::Instant::now();
            let tensors = self.graph.node_weight_mut(*node).unwrap().process(srcs);
            let elapsed = now.elapsed();
            println!(
                "{:.>1$}",
                format_duration(&elapsed).bold(),
                width
                    .saturating_sub(op_name.len())
                    .saturating_sub(shapes_string.len()),
            );
            for (i, tensor) in tensors.into_iter().enumerate() {
                self.tensors.insert((*node, i as u8), tensor);
            }
            let timed_op_name = format!("{:?}", self.node_weight(*node).unwrap());
            if let Some(t) = op_times.get_mut(&timed_op_name) {
                *t += elapsed;
            } else {
                op_times.insert(timed_op_name, elapsed);
            }

            // Check if we can delete the source tensors now
            for (id, ind, _) in src_ids {
                *consumers.get_mut(&(*id, *ind)).unwrap() -= 1;
            }
        }

        // Print out total times
        println!();
        println!(
            "{:->2$} Total Times {:->2$}",
            "",
            "",
            (width.saturating_sub(" Total Times ".len())) / 2
        );
        for (name, elapsed) in op_times.into_iter().sorted_by(|(_, a), (_, b)| b.cmp(a)) {
            print!("{}", name.bold().bright_green());
            println!(
                "{:.>1$}",
                format_duration(&elapsed).bold(),
                width.saturating_sub(name.len()),
            );
        }
        println!("Total: {}", format_duration(&start.elapsed()).bold());
        self.reset();
    }
}

impl Deref for Graph {
    type Target = MainGraph;
    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl DerefMut for Graph {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

/// Get source tensor array for a node
fn get_source_tensors<'a>(
    no_delete: &'a FxHashSet<NodeIndex>,
    tensors: &mut FxHashMap<(NodeIndex, u8), Tensor>,
    // tensors: *mut FxHashMap<(NodeIndex, u8), Tensor>,
    src_ids: &'a [(NodeIndex, u8, ShapeTracker)],
    consumers: &'a FxHashMap<(NodeIndex, u8), usize>,
) -> Vec<(InputTensor, ShapeTracker)> {
    let mut srcs = vec![];
    for (id, ind, sh) in src_ids {
        let id = &(*id, *ind);
        if consumers[id] == 1 && !no_delete.contains(&id.0) {
            let tensor = tensors.remove(id).unwrap();
            debug_assert!(tensor.is_owned());
            srcs.push((InputTensor::new(tensor), *sh));
        } else {
            let tensor = tensors.get(id).unwrap();
            srcs.push((InputTensor::new(tensor.clone()), *sh));
        }
    }
    srcs
}
