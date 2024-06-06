use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use petgraph::{
    algo::toposort,
    stable_graph::{NodeIndex, StableGraph},
    visit::EdgeRef,
    Direction,
};

use crate::{
    op::{Add, Constant, ConstantValue, Function, MaxReduce, Mul, Operator, Recip, SumReduce},
    prelude::*,
};

/// Generic platform-agnostic optimizations. It's a good idea to use these all the time.
pub type GenericCompiler = (
    //RemoveSingleReductions,
    RemoveUnusedNodes,
    ArithmeticElimination,
    CSE,
);

/// [Common subexpression elimination](https://en.wikipedia.org/wiki/Common_subexpression_elimination)
#[derive(Default, Debug)]
pub struct CSE;

impl Compiler for CSE {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &GraphWrapper, mut ids: T) {
        // Look for nodes that have the exact same srcs
        // Loop cause I'm lazy
        let mut eliminated = true;
        while eliminated {
            eliminated = false;
            let mut srcs_set: HashMap<Vec<NodeIndex>, Vec<NodeIndex>> = HashMap::new();
            let nodes = graph.borrow().graph.node_indices().collect_vec();
            for node in nodes {
                let graph_ref = graph.borrow();

                if graph_ref
                    .graph
                    .node_weight(node)
                    .unwrap()
                    .as_any()
                    .is::<Function>()
                {
                    continue;
                }
                let srcs = graph_ref
                    .graph
                    .edges_directed(node, petgraph::Direction::Incoming)
                    .filter(|e| !e.weight().is_schedule())
                    .sorted_by_key(|e| e.weight().as_data().unwrap().0)
                    .map(|e| e.source())
                    .collect_vec();

                if let Some(other_nodes) = srcs_set.get(&srcs) {
                    drop(graph_ref);
                    for other_node in other_nodes {
                        let graph_ref = graph.borrow();
                        let a = graph_ref.graph.node_weight(node).unwrap();
                        let Some(b) = graph_ref.graph.node_weight(*other_node) else {
                            continue;
                        };
                        if format!("{a:?}") != format!("{b:?}") {
                            // Sloppy way to check if ops are equal, but we only expect primops here so it's ok
                            continue;
                        }
                        let a_src_shapes = graph_ref
                            .get_sources(node)
                            .into_iter()
                            .map(|(_, _, a)| a)
                            .collect_vec();
                        let b_src_shapes = graph_ref
                            .get_sources(*other_node)
                            .into_iter()
                            .map(|(_, _, a)| a)
                            .collect_vec();
                        drop(graph_ref);
                        if a_src_shapes != b_src_shapes {
                            continue;
                        }
                        let mut graph_mut = graph.borrow_mut();
                        // If the op, input shapes, and output shape is the same, we can combine them (UNCLEAR IF THIS IS TRUE, NEED PROPER PartialEq)
                        // Carry over outgoing edges from node to other_node
                        move_outgoing_edge(node, *other_node, &mut graph_mut.graph);
                        // Transfer all references to node over to other node
                        remap(node, *other_node, &mut ids, &mut graph_mut);
                        // Remove node
                        graph_mut.graph.remove_node(node);
                        eliminated = true;
                        break;
                    }
                    if eliminated {
                        break;
                    }
                }
                if let Some(nodes) = srcs_set.get_mut(&srcs) {
                    nodes.push(node);
                } else {
                    srcs_set.insert(srcs, vec![node]);
                }
            }
            srcs_set.clear();
        }
    }
}

/// Remove maxreduces and sumreduces that don't do anything
#[derive(Default)]
pub struct RemoveSingleReductions;

impl Compiler for RemoveSingleReductions {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &GraphWrapper, mut ids: T) {
        let nodes = graph.borrow().graph.node_indices().collect::<Vec<_>>();
        for node in nodes {
            let graph_ref = graph.borrow();
            let dim = if let Some(red) = graph_ref
                .graph
                .node_weight(node)
                .unwrap()
                .as_any()
                .downcast_ref::<SumReduce>()
            {
                Some(red.0)
            } else {
                graph_ref
                    .graph
                    .node_weight(node)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<MaxReduce>()
                    .map(|red| red.0)
            };
            if let Some(dim) = dim {
                if graph_ref
                    .graph
                    .edges_directed(node, Direction::Incoming)
                    .next()
                    .map(|e| {
                        e.weight()
                            .as_data()
                            .map(|w| {
                                w.2.dims[w.2.indexes[dim]]
                                    .to_usize()
                                    .map(|i| i == 1)
                                    .unwrap_or_default()
                            })
                            .unwrap_or_default()
                    })
                    .unwrap_or_default()
                {
                    let upstream = graph_ref
                        .neighbors_directed(node, Direction::Incoming)
                        .next()
                        .unwrap();
                    drop(graph_ref);
                    let mut graph_mut = graph.borrow_mut();
                    remap(node, upstream, &mut ids, &mut graph_mut);
                    move_outgoing_edge(node, upstream, &mut graph_mut.graph);
                    graph_mut.graph.remove_node(node);
                }
            }
        }
    }
}

/// Remove unused nodes
#[derive(Default, Debug)]
pub struct RemoveUnusedNodes;

impl Compiler for RemoveUnusedNodes {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &GraphWrapper, _: T) {
        // Reverse topo sort
        let nodes = toposort(&graph.borrow().graph, None);
        for node in nodes.unwrap().into_iter().rev() {
            let graph_ref = graph.borrow();
            let edges_directed = graph_ref.edges_directed(node, Direction::Outgoing).count();
            let contains_node = graph_ref.no_delete.contains(&node);
            drop(graph_ref);
            if edges_directed == 0 && !contains_node {
                // No dependencies and not marked for no_delete, so remove
                graph.borrow_mut().remove_node(node);
            }
        }
    }
}

/// Enforce the graph gets ran in strictly depth-first order
#[derive(Default, Debug)]
pub struct DepthFirst;

impl Compiler for DepthFirst {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &GraphWrapper, _: T) {
        fn toposort(
            id: NodeIndex,
            graph: &StableGraph<Box<dyn Operator>, Dependency>,
            visited: &mut HashSet<NodeIndex>,
        ) -> (Vec<NodeIndex>, usize, bool) {
            if visited.contains(&id) {
                return (vec![], 0, false);
            }
            // Loop through node sources
            let stacks = graph
                .edges_directed(id, Direction::Incoming)
                .sorted_by_key(|e| e.source())
                .map(|e| toposort(e.source(), graph, visited))
                .collect::<Vec<_>>();
            let num_stacks = stacks.len();

            let mut final_stack = vec![];
            let mut complete = true;
            for (mut stack, _, c) in stacks.into_iter().sorted_by_key(|(_, _, b)| !*b) {
                final_stack.append(&mut stack);
                complete &= c;
            }
            final_stack.push(id);
            visited.insert(id);

            (final_stack, num_stacks, complete)
        }

        // Depth-first toposort
        let graph_ref = graph.borrow();
        let mut visited = HashSet::default();
        let mut pre_sorted = petgraph::algo::toposort(&graph_ref.graph, None).unwrap();
        pre_sorted.reverse();
        let mut stacks = vec![];
        for node in pre_sorted {
            if !visited.contains(&node) {
                stacks.push(toposort(node, &graph_ref.graph, &mut visited));
            }
        }
        let mut nodes = vec![];
        for (mut stack, _, _) in stacks.into_iter().sorted_by_key(|(_, _, b)| !*b) {
            nodes.append(&mut stack);
        }

        // Insert schedule deps
        drop(graph_ref);
        let mut graph_mut = graph.borrow_mut();
        for i in 0..nodes.len() - 1 {
            graph_mut.add_schedule_dependency(nodes[i], nodes[i + 1]);
        }
    }
}

/// **Reduces arithmetic expressions**
///
/// - Current: x + 0 => x, x * 1 => x
/// - TODO: x / x => 1, x - x => 0, x * 0 => 0, x - 0 => x, x * 0 => 0, 0 / x => 0
/// - TODO: Find a much cleaner way to do these eliminations
#[derive(Debug, Default)]
pub struct ArithmeticElimination;

impl Compiler for ArithmeticElimination {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &GraphWrapper, mut ids: T) {
        // x + 0, 0 + x
        let zero = constant(0.);
        let inp = node();
        let add1 = binary::<Add>(zero.clone(), inp.clone());
        let add2 = binary::<Add>(inp.clone(), zero.clone());
        let mut s1 = add1.clone().search(graph);
        let mut s2 = add2.clone().search(graph);
        while s1.next_match() || s2.next_match() {
            let (inp, zero, add) = if s1.matched {
                (s1.get(&inp), s1.get(&zero), s1.get(&add1))
            } else {
                (s2.get(&inp), s2.get(&zero), s2.get(&add2))
            };
            let graph_ref = graph.borrow();
            if graph_ref.no_delete.contains(&zero) {
                continue;
            }
            // Carry over outgoing edges
            let input_shape = graph_ref
                .graph
                .edges_connecting(inp, add)
                .find_map(|e| e.weight().as_data())
                .unwrap()
                .2;
            if input_shape.is_reshaped() {
                // If any output shape is non-contiguous, we need to keep the op for it's contiguous functionality TODO: replace with explicit contiguous op here
                if graph_ref
                    .graph
                    .edges_connecting(inp, add)
                    .filter_map(|e| e.weight().as_data())
                    .any(|(_, _, sh)| sh.is_reshaped())
                {
                    continue;
                }
                let elems = graph_ref
                    .graph
                    .edges_directed(add, petgraph::Direction::Outgoing)
                    .map(|e| (*e.weight(), e.target()))
                    .collect::<Vec<_>>();
                drop(graph_ref);
                for (weight, target) in elems {
                    if let Some(weight) = weight.as_data() {
                        graph.borrow_mut().graph.add_edge(
                            inp,
                            target,
                            Dependency::Data {
                                input_order: weight.0,
                                output_order: weight.1,
                                shape: input_shape,
                            },
                        );
                    }
                }
            } else {
                drop(graph_ref);
                move_outgoing_edge(add, inp, &mut graph.borrow_mut().graph);
            }
            let mut graph_mut = graph.borrow_mut();
            remap(add, inp, &mut ids, &mut graph_mut);
            if graph_mut
                .graph
                .edges_directed(zero, Direction::Outgoing)
                .count()
                == 1
            {
                graph_mut.graph.remove_node(zero);
            }
            graph_mut.graph.remove_node(add);
        }
        // x * 1, 1 * x
        let one = constant(1.);
        let inp = node();
        let mul1 = binary::<Mul>(one.clone(), inp.clone());
        let mul2 = binary::<Mul>(inp.clone(), one.clone());
        let mut s1 = mul1.clone().search(graph);
        let mut s2 = mul2.clone().search(graph);
        while s1.next_match() || s2.next_match() {
            let (inp, one, mul) = if s1.matched {
                (s1.get(&inp), s1.get(&one), s1.get(&mul1))
            } else {
                (s2.get(&inp), s2.get(&one), s2.get(&mul2))
            };
            let graph_ref = graph.borrow();
            if graph_ref.no_delete.contains(&one) {
                continue;
            }
            // Carry over outgoing edges
            let input_shape = graph_ref
                .graph
                .edges_connecting(inp, mul)
                .find_map(|e| e.weight().as_data())
                .unwrap()
                .2;
            if input_shape.is_reshaped() {
                // If any output shape is non-contiguous, we need to keep the op for it's contiguous functionality TODO: replace with explicit contiguous op here
                if graph_ref
                    .graph
                    .edges_connecting(inp, mul)
                    .filter_map(|e| e.weight().as_data())
                    .any(|(_, _, sh)| sh.is_reshaped())
                {
                    continue;
                }
                let elems = graph_ref
                    .graph
                    .edges_directed(mul, petgraph::Direction::Outgoing)
                    .map(|e| (*e.weight(), e.target()))
                    .collect::<Vec<_>>();
                drop(graph_ref);
                for (weight, target) in elems {
                    if let Some(weight) = weight.as_data() {
                        graph.borrow_mut().graph.add_edge(
                            inp,
                            target,
                            Dependency::Data {
                                input_order: weight.0,
                                output_order: weight.1,
                                shape: input_shape,
                            },
                        );
                    }
                }
            } else {
                drop(graph_ref);
                move_outgoing_edge(mul, inp, &mut graph.borrow_mut().graph);
            }
            let mut graph_mut = graph.borrow_mut();
            remap(mul, inp, &mut ids, &mut graph_mut);
            graph_mut.safe_remove_node(one, 1);
            graph_mut.graph.remove_node(mul);
        }
        // recip(recip(x))
        let inp = node();
        let intermediate = unary::<Recip>(inp.clone());
        let out = unary::<Recip>(intermediate.clone());
        let mut s = out.clone().search(graph);
        while s.next_match() {
            let (inp, intermediate, out) = (s.get(&inp), s.get(&intermediate), s.get(&out));
            let graph_ref = graph.borrow();
            if graph_ref.no_delete.contains(&intermediate) {
                continue;
            }
            // Carry over outgoing edges
            let input_shape = graph_ref
                .graph
                .edges_connecting(inp, intermediate)
                .find_map(|e| e.weight().as_data())
                .unwrap()
                .2;
            if input_shape.is_reshaped() {
                // If any output shape is non-contiguous, we need to keep the op for it's contiguous functionality TODO: replace with explicit contiguous op here
                if graph_ref
                    .graph
                    .edges_connecting(inp, intermediate)
                    .filter_map(|e| e.weight().as_data())
                    .any(|(_, _, sh)| sh.is_reshaped())
                    || graph_ref
                        .graph
                        .edges_connecting(intermediate, out)
                        .filter_map(|e| e.weight().as_data())
                        .any(|(_, _, sh)| sh.is_reshaped())
                {
                    continue;
                }
                let elems = graph_ref
                    .graph
                    .edges_directed(intermediate, petgraph::Direction::Outgoing)
                    .map(|e| (*e.weight(), e.target()))
                    .collect::<Vec<_>>();
                drop(graph_ref);
                for (weight, target) in elems {
                    if let Some(weight) = weight.as_data() {
                        graph.borrow_mut().graph.add_edge(
                            inp,
                            target,
                            Dependency::Data {
                                input_order: weight.0,
                                output_order: weight.1,
                                shape: input_shape,
                            },
                        );
                    }
                }
            } else {
                drop(graph_ref);
                move_outgoing_edge(out, inp, &mut graph.borrow_mut().graph);
            }
            let mut graph_mut = graph.borrow_mut();
            remap(intermediate, inp, &mut ids, &mut graph_mut);
            remap(out, inp, &mut ids, &mut graph_mut);
            graph_mut.remove_node(out);
            graph_mut.safe_remove_node(intermediate, 0);
        }
    }
}

fn constant(num: f32) -> SelectGraph {
    let mut n = op::<Constant>();
    n.check(move |o, _| {
        if let Some(Constant(ConstantValue::Float(f), _)) = o.as_any().downcast_ref::<Constant>() {
            *f == num
        } else {
            false
        }
    });
    n
}
