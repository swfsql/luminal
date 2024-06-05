use std::{cell::RefCell, marker::PhantomData, rc::Rc, sync::Arc};

use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};
use itertools::Itertools;
use luminal::prelude::{petgraph::visit::EdgeRef, *};
use rustc_hash::FxHashMap;

use crate::{
    binary::CudaSub,
    compile_and_load_kernel, constant,
    prim::{CudaAdd, CudaContiguous, CudaCopyFromDevice, CudaCopyToDevice, CudaSumReduce},
    CudaData, CudaFloat,
};

#[derive(Clone)]
pub struct CudaARange<T> {
    function: CudaFunction,
    device: Arc<CudaDevice>,
    pub size: BigExpression,
    dyn_map: Rc<RefCell<FxHashMap<char, usize>>>,
    _phantom: PhantomData<T>,
}
crate::debug_type!(CudaARange);

impl<T: CudaFloat> CudaARange<T> {
    pub fn new(
        device: Arc<CudaDevice>,
        size: BigExpression,
        dyn_map: Rc<RefCell<FxHashMap<char, usize>>>,
    ) -> Self {
        let type_name = T::type_name();
        let code = format!(
            "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, int n_elements) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {{
        out[idx] = ({type_name})idx;
    }}
}}"
        );
        Self {
            function: compile_and_load_kernel(code, &device),
            device,
            size,
            _phantom: Default::default(),
            dyn_map,
        }
    }
}

impl<T: CudaFloat> Operator for CudaARange<T> {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let n_elements = self.size.exec(&self.dyn_map.borrow()).unwrap();
        let mut out = self.device.alloc_zeros::<T>(n_elements).unwrap();
        unsafe {
            self.function
                .clone()
                .launch(
                    LaunchConfig::for_num_elems(n_elements as u32),
                    (&mut out, n_elements as i32),
                )
                .unwrap();
        }

        vec![Tensor::new(CudaData(out))]
    }
}

#[derive(Debug, Default)]
pub struct ARangeCompiler<T: CudaFloat>(PhantomData<T>);

impl<T: CudaFloat> Compiler for ARangeCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &GraphWrapper, _: To) {
        let dev = CudaDevice::new(0).unwrap();
        // TODO: Make sure this actually checks the shape transformations to ensure pooling happens
        let contig_one = constant::<T>(1.);
        let contig1 = unary::<CudaContiguous<T>>(contig_one.clone());
        let sum_reduce =
            unary::<CudaSumReduce<T>>(unary::<CudaContiguous<T>>(unary::<CudaContiguous<T>>(
                unary::<CudaContiguous<T>>(contig1.clone()),
            )));
        let sub = binary::<CudaSub<T>>(sum_reduce.clone(), constant::<T>(1.));
        let mut s1 = sub.clone().search(graph);
        let neg_one = constant::<T>(-1.);
        let add = binary::<CudaAdd<T>>(sum_reduce, neg_one.clone());
        let mut s2 = add.clone().search(graph);

        while s1.next_match() || s2.next_match() {
            let graph_ref = graph.borrow();
            let s = if s1.matched { &s1 } else { &s2 };
            let arange_amount = {
                let sh = graph_ref
                    .edges_connecting(s.get(&contig_one), s.get(&contig1))
                    .next()
                    .unwrap()
                    .weight()
                    .as_data()
                    .unwrap()
                    .2;
                sh.dims[sh.indexes[sh.len() - 1]]
            };
            let dyn_map = graph_ref.dyn_map.clone();
            drop(graph_ref);
            let arange_op = graph
                .add_op(CudaARange::<T>::new(
                    dev.clone(),
                    arange_amount.into(),
                    dyn_map,
                ))
                .finish();
            let fin = if s1.matched {
                s1.get(&sub)
            } else {
                s2.get(&add)
            };
            let mut graph_mut = graph.borrow_mut();
            move_outgoing_edge(fin, arange_op, &mut graph_mut.graph);
            graph_mut.remove_node(fin);
            drop(graph_mut);
            s.try_delete();
        }
    }
}

// Sometimes CopyTo -> CopyFrom and CopyFrom -> CopyTo patterns remain, so let's clean them up
#[derive(Debug, Default)]
pub struct CopyCompiler<T>(PhantomData<T>);

impl<T: CudaFloat> Compiler for CopyCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &GraphWrapper, mut ids: To) {
        let graph_ref = graph.borrow();
        let nodes = graph_ref
            .edge_indices()
            .filter_map(|e| graph_ref.edge_endpoints(e))
            .filter(|(a, b)| {
                (graph_ref
                    .node_weight(*a)
                    .unwrap()
                    .as_any()
                    .is::<CudaCopyToDevice<T>>()
                    && graph_ref
                        .node_weight(*b)
                        .unwrap()
                        .as_any()
                        .is::<CudaCopyFromDevice<T>>())
                    || (graph_ref
                        .node_weight(*a)
                        .unwrap()
                        .as_any()
                        .is::<CudaCopyFromDevice<T>>()
                        && graph_ref
                            .node_weight(*b)
                            .unwrap()
                            .as_any()
                            .is::<CudaCopyToDevice<T>>())
            })
            .unique_by(|n| n.0)
            .unique_by(|n| n.1)
            .collect::<Vec<_>>();
        for (first, second) in nodes {
            let graph_ref = graph.borrow();
            if graph_ref
                .edges_directed(first, petgraph::Direction::Outgoing)
                .filter(|e| graph_ref.contains_node(e.target()))
                .filter(|e| {
                    !graph_ref
                        .node_weight(e.target())
                        .unwrap()
                        .as_any()
                        .is::<CudaCopyFromDevice<T>>()
                        && !graph_ref
                            .node_weight(e.target())
                            .unwrap()
                            .as_any()
                            .is::<CudaCopyToDevice<T>>()
                })
                .count()
                > 0
                || graph_ref.no_delete.contains(&first)
            {
                continue;
            }
            let source = graph_ref.get_sources(first)[0];
            drop(graph_ref);
            let mut graph_mut = graph.borrow_mut();
            move_outgoing_edge(second, source.0, &mut graph_mut);
            remap(second, source.0, &mut ids, &mut graph_mut);
            graph_mut.remove_node(second);
            for dest in graph_mut
                .get_dests(first)
                .iter()
                .map(|(i, _)| *i)
                .collect::<Vec<_>>()
            {
                move_outgoing_edge(dest, source.0, &mut graph_mut);
                remap(dest, source.0, &mut ids, &mut graph_mut);
                graph_mut.remove_node(dest);
            }
            graph_mut.remove_node(first);
        }
    }
}
