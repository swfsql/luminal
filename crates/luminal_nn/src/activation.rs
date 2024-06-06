use luminal::prelude::*;

/// Rectified Linear Unit activation function
pub struct ReLU;

impl InitModule for ReLU {
    fn initialize(_: &GraphWrapper) -> Self {
        Self
    }
}

impl SerializeModule for ReLU {
    fn serialize(&self, _: &mut Serializer) {}
}

impl<S: Shape> Module<GraphTensor<S>> for ReLU {
    type Output = GraphTensor<S>;

    fn forward(&self, input: GraphTensor<S>) -> Self::Output {
        input.relu()
    }
}

/// Sigmoid activation function
pub struct Sigmoid;

impl InitModule for Sigmoid {
    fn initialize(_: &GraphWrapper) -> Self {
        Self
    }
}

impl SerializeModule for Sigmoid {
    fn serialize(&self, _: &mut Serializer) {}
}

impl<S: ConstShape> Module<GraphTensor<S>> for Sigmoid {
    type Output = GraphTensor<S>;

    fn forward(&self, input: GraphTensor<S>) -> Self::Output {
        input.sigmoid()
    }
}

/// Swish activation function
pub struct Swish;

impl InitModule for Swish {
    fn initialize(_: &GraphWrapper) -> Self {
        Self
    }
}

impl SerializeModule for Swish {
    fn serialize(&self, _: &mut Serializer) {}
}

impl<S: ConstShape> Module<GraphTensor<S>> for Swish {
    type Output = GraphTensor<S>;

    fn forward(&self, input: GraphTensor<S>) -> Self::Output {
        input.swish()
    }
}

/// Tanh activation function
pub struct Tanh;

impl InitModule for Tanh {
    fn initialize(_: &GraphWrapper) -> Self {
        Self
    }
}

impl SerializeModule for Tanh {
    fn serialize(&self, _: &mut Serializer) {}
}

impl<S: ConstShape> Module<GraphTensor<S>> for Tanh {
    type Output = GraphTensor<S>;

    fn forward(&self, input: GraphTensor<S>) -> Self::Output {
        input.tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::ReLU;
    use crate::Linear;
    use dfdx::prelude::{Module as DfdxModule, *};
    use luminal::{
        prelude::{Module, *},
        tests::assert_close,
    };

    #[test]
    fn test_relu_and_linear() {
        // Test single and batch, unoptimized and optimized
        let cx = Graph::new();
        let batch = cx
            .tensor::<R2<2, 3>>()
            .set(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let a = cx.tensor::<R1<3>>().set(vec![1.0, 2.0, 3.0]);

        let model: (Linear<3, 4>, ReLU, Linear<4, 2>) = InitModule::initialize(&cx);
        model
            .0
            .weight
            .clone()
            .set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]);
        model
            .2
            .weight
            .clone()
            .set(vec![1., 2., 3., 1., 2., 3., 1., 2.]);
        let mut b = model.forward(a).retrieve();
        let mut batch_out = model.forward(batch).retrieve();

        cx.execute();

        let unoptimized_b = b.data();
        let unoptimized_batch_out = batch_out.data();

        cx.compile(GenericCompiler::default(), (&mut b, &mut batch_out));
        cx.execute();

        assert_close(&unoptimized_b, &b.data());
        assert_close(&unoptimized_batch_out, &batch_out.data());

        // Test against dfdx
        let dev = Cpu::default();
        let mut model = <(
            dfdx::nn::modules::builders::UnbiasedLinear<3, 4>,
            dfdx::nn::modules::builders::ReLU,
            dfdx::nn::modules::builders::UnbiasedLinear<4, 2>,
        )>::build_on_device(&dev);
        // Set weights
        model.0.weight = dev
            .tensor_from_vec(
                vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<4>),
            )
            .permute();
        model.2.weight = dev
            .tensor_from_vec(
                vec![1., 2., 3., 1., 2., 3., 1., 2.],
                (dfdx::shapes::Const::<4>, dfdx::shapes::Const::<2>),
            )
            .permute();
        let a = dev.tensor_from_vec(vec![1.0, 2.0, 3.0], (dfdx::shapes::Const::<3>,));
        let d_batch = dev.tensor_from_vec(
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<3>),
        );
        let out = model.forward(a);
        let d_batch_out = model.forward(d_batch);

        assert_close(&unoptimized_b, &out.as_vec());
        assert_close(&unoptimized_batch_out, &d_batch_out.as_vec());
    }
}
