use crate::{Linear, ReLU, Repeated};
use luminal::prelude::*;

use super::attention::MultiHeadSelfAttention;

/// A transformer encoder as layed out in [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).
pub type TransformerEncoder<
    const DIM: usize,
    const FF: usize,
    const HEADS: usize,
    const LAYERS: usize,
> = Repeated<TransformerEncoderBlock<DIM, FF, HEADS>, LAYERS>;

/// A single transformer encoder block
pub struct TransformerEncoderBlock<const DIM: usize, const FF: usize, const HEADS: usize> {
    pub attention: MultiHeadSelfAttention<DIM, DIM, DIM, HEADS>,
    pub ff: (Linear<DIM, FF>, ReLU, Linear<FF, DIM>),
}

impl<const DIM: usize, const FF: usize, const HEADS: usize> InitModule
    for TransformerEncoderBlock<DIM, FF, HEADS>
{
    fn initialize(cx: &GraphWrapper) -> Self {
        Self {
            attention: InitModule::initialize(cx),
            ff: InitModule::initialize(cx),
        }
    }
}

impl<const DIM: usize, const FF: usize, const HEADS: usize> SerializeModule
    for TransformerEncoderBlock<DIM, FF, HEADS>
{
    fn serialize(&self, s: &mut Serializer) {
        s.module("self_attn", &self.attention);
        s.module("ff", &self.ff);
    }
}

// Single
impl<const DIM: usize, const FF: usize, const HEADS: usize, S: Dimension>
    Module<GraphTensor<(S, Const<DIM>)>> for TransformerEncoderBlock<DIM, FF, HEADS>
{
    type Output = GraphTensor<(S, Const<DIM>)>;

    fn forward(&self, input: GraphTensor<(S, Const<DIM>)>) -> Self::Output {
        // Pass to batched forward
        <Self as Module<GraphTensor<(Const<1>, S, Const<DIM>)>>>::forward(self, input.expand())
            .reshape()
    }
}

// Batched
impl<const DIM: usize, const FF: usize, const HEADS: usize, S: Dimension, B: Dimension>
    Module<GraphTensor<(B, S, Const<DIM>)>> for TransformerEncoderBlock<DIM, FF, HEADS>
{
    type Output = GraphTensor<(B, S, Const<DIM>)>;

    fn forward(&self, x: GraphTensor<(B, S, Const<DIM>)>) -> Self::Output {
        let x = x.clone() + self.attention.forward(x);
        let x = x.layer_norm::<Axis<2>, _>(1e-5);
        let x = x.clone() + self.ff.forward(x);
        x.layer_norm::<Axis<2>, _>(1e-5)
    }
}

#[cfg(test)]
mod tests {
    use dfdx::{
        prelude::{DeviceBuildExt, Module as DfdxModule},
        tensor::{Cpu, TensorFromVec},
        tensor_ops::PermuteTo,
    };

    use luminal::{
        prelude::{Module, *},
        tests::assert_close,
    };

    use super::TransformerEncoderBlock;
    #[test]
    fn test_transformer_encoder_block() {
        let cx = Graph::new();
        let model: TransformerEncoderBlock<3, 4, 1> = InitModule::initialize(&cx);
        model
            .attention
            .w_k
            .weight
            .clone()
            .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
        model
            .attention
            .w_q
            .weight
            .clone()
            .set(vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.]);
        model
            .attention
            .w_v
            .weight
            .clone()
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.]);
        model
            .attention
            .w_o
            .weight
            .clone()
            .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
        model
            .ff
            .0
            .weight
            .clone()
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 11., 2., 3.]);
        model
            .ff
            .2
            .weight
            .clone()
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 3., -1., 2.]);

        let a = cx
            .tensor::<(Dyn<'s'>, luminal::shape::Const<3>)>()
            .set_dyn(vec![-1., 2., 3., 3., 3., -1.], &[2, 3]);
        let b = model.forward(a).retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let mut d_model: dfdx::nn::modules::TransformerEncoderBlock<3, 1, 4, f32, Cpu> =
            d_dev
                .build_module::<dfdx::nn::modules::builders::TransformerEncoderBlock<3, 1, 4>, f32>(
                );
        d_model.self_attn.w_k.bias.copy_from(&[0.0, 0.0, 0.0]);
        d_model.self_attn.w_v.bias.copy_from(&[0.0, 0.0, 0.0]);
        d_model.self_attn.w_q.bias.copy_from(&[0.0, 0.0, 0.0]);
        d_model.self_attn.w_o.bias.copy_from(&[0., 0., 0.]);
        d_model.self_attn.w_o.weight = d_dev
            .tensor_from_vec(
                vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.self_attn.w_k.weight = d_dev
            .tensor_from_vec(
                vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.self_attn.w_q.weight = d_dev
            .tensor_from_vec(
                vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.self_attn.w_v.weight = d_dev
            .tensor_from_vec(
                vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.ff.0 .0.weight = d_dev
            .tensor_from_vec(
                vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 11., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<4>),
            )
            .permute();
        d_model.ff.0 .0.bias =
            d_dev.tensor_from_vec(vec![0., 0., 0., 0.], (dfdx::shapes::Const::<4>,));
        d_model.ff.0 .2.weight = d_dev
            .tensor_from_vec(
                vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 3., -1., 2.],
                (dfdx::shapes::Const::<4>, dfdx::shapes::Const::<3>),
            )
            .permute();
        d_model.ff.0 .2.bias = d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
        d_model.norm1.gamma = d_dev.tensor_from_vec(vec![1., 1., 1.], (dfdx::shapes::Const::<3>,));
        d_model.norm2.gamma = d_dev.tensor_from_vec(vec![1., 1., 1.], (dfdx::shapes::Const::<3>,));
        d_model.norm1.epsilon = 1e-5;
        d_model.norm2.beta = d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
        d_model.norm1.beta = d_dev.tensor_from_vec(vec![0., 0., 0.], (dfdx::shapes::Const::<3>,));
        d_model.norm2.epsilon = 1e-5;
        let d_a = d_dev.tensor_from_vec(
            vec![-1., 2., 3., 3., 3., -1.],
            (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<3>),
        );
        let d_b = d_model.forward(d_a);

        assert_close(&b.data(), &d_b.as_vec());
    }
}
