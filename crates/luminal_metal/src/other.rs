use std::{any::Any, marker::PhantomData, sync::Arc};

use luminal::{
    op::{InputTensor, Operator},
    prelude::{petgraph::visit::EdgeRef, *},
};
use metal_rs::{
    objc::rc::autoreleasepool, Buffer, CommandBufferRef, CommandQueue, ComputePassDescriptor,
    ComputePipelineState, Device, MTLResourceOptions,
};
use rustc_hash::FxHashMap;

use crate::{
    compile_function, constant,
    prim::{MetalAdd, MetalContiguous, MetalCopyFromDevice, MetalCopyToDevice, MetalSumReduce},
    DispatchNElements, MetalBuffer, MetalFloat, MetalKernel, MetalKernelWrapper, SetInt,
};

use super::binary::MetalSub;

/// Sometimes CopyTo -> CopyFrom and CopyFrom -> CopyTo patterns remain, so let's clean them up
#[derive(Debug, Default)]
pub struct CopyCompiler<T>(PhantomData<T>);

impl<T: MetalFloat> Compiler for CopyCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &GraphWrapper, mut ids: To) {
        let first = op::<MetalCopyToDevice<T>>();
        let second = op::<MetalCopyFromDevice<T>>();
        let mut s = first.clone().connect(second.clone()).search(graph);
        while s.next_match() {
            let (first, second) = (s.get(&first), s.get(&second));
            // Ensure there are no dests from first that are not copies
            if graph
                .edges_directed(first, petgraph::Direction::Outgoing)
                .filter(|e| {
                    let target = graph.node_weight(e.target()).unwrap().as_any();
                    !target.is::<MetalCopyFromDevice<T>>() && !target.is::<MetalCopyToDevice<T>>()
                })
                .count()
                > 0
                || graph.no_delete.contains(&first)
            {
                continue;
            }
            let Some((source, _, _)) = graph.get_sources(first).pop() else {
                continue;
            };
            move_outgoing_edge(second, source, graph);
            remap(second, source, &mut ids, graph);
            graph.remove_node(second);
            for dest in graph
                .get_dests(first)
                .iter()
                .map(|(i, _)| *i)
                .collect::<Vec<_>>()
            {
                move_outgoing_edge(dest, source, graph);
                remap(dest, source, &mut ids, graph);
                graph.remove_node(dest);
            }
            graph.remove_node(first);
            s.clear_cached_results();
        }
    }
}

/// Special kernel for producing aranges
#[derive(Clone)]
pub struct MetalARange<T: MetalFloat> {
    pipeline: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
    pub size: BigExpression,
    dyn_map: *const FxHashMap<char, usize>,
    _phantom: PhantomData<T>,
}

impl<T: MetalFloat> std::fmt::Debug for MetalARange<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalARange({:?})", self.size)
    }
}

impl<T: MetalFloat> MetalARange<T> {
    pub fn new(
        device: Device,
        queue: CommandQueue,
        size: BigExpression,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let type_name = T::type_name();
        Self {
            pipeline: compile_function("metal_arange", &format!("
#include <metal_stdlib>
using namespace metal;
kernel void metal_arange(device {type_name} *out [[buffer(0)]], device int& n_elements [[buffer(1)]], uint idx [[thread_position_in_grid]]) {{
    if (idx < n_elements) {{
        out[idx] = ({type_name})idx;
    }}
}}"), &device),
            queue,
            device,
            size,
            dyn_map,
            _phantom: Default::default(),
        }
    }
}

impl<T: MetalFloat> MetalKernel for MetalARange<T> {
    fn output_buffer_sizes(&self, _: &[ShapeTracker]) -> Vec<BigExpression> {
        vec![self.size.clone() * std::mem::size_of::<f16>()]
    }
    fn metal_forward(
        &self,
        _: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        _: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        // Calculate size
        let size = self
            .size
            .exec(unsafe { self.dyn_map.as_ref().unwrap() })
            .unwrap();

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Set inputs
        encoder.set_buffer(0, Some(output_buffers[0]), 0);
        encoder.set_u32(1, size as u32);

        // Execute
        encoder.dispatch_1d(size);
        encoder.end_encoding();
    }
}

impl<T: MetalFloat> Operator for MetalARange<T> {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        autoreleasepool(|| {
            // Set up command buffer and output buffer
            let command_buffer = self.queue.new_command_buffer();
            let size = self
                .size
                .exec(unsafe { self.dyn_map.as_ref().unwrap() })
                .unwrap();
            let out = self.device.new_buffer(
                (size * std::mem::size_of::<f16>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            self.metal_forward(&[], command_buffer, &[], &[&out]);

            command_buffer.commit();
            command_buffer.wait_until_completed();

            vec![Tensor::new(MetalBuffer(out))]
        })
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "metal" {
            #[allow(clippy::arc_with_non_send_sync)]
            return Some(Box::new(MetalKernelWrapper(Arc::new(Box::new(
                self.clone(),
            )))));
        }
        None
    }
}

/// Replace the arange pattern with a special kernel. This must be ran **after** the subtraction compiler
#[derive(Default, Debug)]
pub struct ARangeCompiler<T: MetalFloat>(PhantomData<T>);

impl<T: MetalFloat> Compiler for ARangeCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &GraphWrapper, _: To) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();

        // TODO: Make sure this actually checks the shape transformations to ensure pooling happens
        let contig_one = constant::<T>(1.);
        let contig1 = unary::<MetalContiguous<T>>(contig_one.clone());
        let sum_reduce =
            unary::<MetalSumReduce<T>>(unary::<MetalContiguous<T>>(unary::<MetalContiguous<T>>(
                unary::<MetalContiguous<T>>(contig1.clone()),
            )));
        let sub = binary::<MetalSub<T>>(sum_reduce.clone(), constant::<T>(1.));
        let mut s1 = sub.clone().search(graph);
        let neg_one = constant::<T>(-1.);
        let add = binary::<MetalAdd<T>>(sum_reduce, neg_one.clone());
        let mut s2 = add.clone().search(graph);

        while s1.next_match() || s2.next_match() {
            let s = if s1.matched { &s1 } else { &s2 };
            let arange_amount = {
                let sh = graph
                    .edges_connecting(s.get(&contig_one), s.get(&contig1))
                    .next()
                    .unwrap()
                    .weight()
                    .as_data()
                    .unwrap()
                    .2;
                sh.dims[sh.indexes[sh.len() - 1]]
            };
            let arange_op = graph
                .add_op(MetalARange::<T>::new(
                    dev.clone(),
                    queue.clone(),
                    arange_amount.into(),
                    &graph.dyn_map,
                ))
                .finish();
            let fin = if s1.matched {
                s1.get(&sub)
            } else {
                s2.get(&add)
            };
            move_outgoing_edge(fin, arange_op, graph);
            graph.remove_node(fin);
            s.try_delete();
        }
    }
}
