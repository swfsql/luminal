use luminal::{prelude::*, tests::random_vec};

pub struct Embedding<const N: usize, const DIM: usize> {
    pub weight: GraphTensor<R2<N, DIM>>,
}

impl<const A: usize, const B: usize> InitModule for Embedding<A, B> {
    fn initialize(cx: &GraphWrapper) -> Self {
        Self {
            weight: cx.named_tensor("Embedding Weight").set(random_vec(A * B)),
        }
    }
}

impl<const A: usize, const B: usize> SerializeModule for Embedding<A, B> {
    fn serialize(&self, s: &mut luminal::module::Serializer) {
        s.tensor("weight", self.weight.clone());
    }
}

// Single
impl<S: Dimension, const N: usize, const DIM: usize> Module<GraphTensor<(S,)>>
    for Embedding<N, DIM>
{
    type Output = GraphTensor<(S, Const<DIM>)>;

    fn forward(&self, input: GraphTensor<(S,)>) -> Self::Output {
        self.weight.clone().gather(input)
    }
}

// Batch
impl<B: Dimension, S: Dimension, const N: usize, const DIM: usize> Module<GraphTensor<(B, S)>>
    for Embedding<N, DIM>
{
    type Output = GraphTensor<(B, S, Const<DIM>)>;

    fn forward(&self, input: GraphTensor<(B, S)>) -> Self::Output {
        self.weight
            .clone()
            .gather(input.dyn_reshape::<(Dyn<'-'>,), _>(&[B::size() * S::size()]))
            .reshape()
    }
}

pub struct PermutedEmbedding<const N: usize, const DIM: usize> {
    pub weight: GraphTensor<R2<DIM, N>>,
}

impl<const A: usize, const B: usize> InitModule for PermutedEmbedding<A, B> {
    fn initialize(cx: &GraphWrapper) -> Self {
        Self {
            weight: cx.named_tensor("Embedding Weight").set(random_vec(A * B)),
        }
    }
}

impl<const A: usize, const B: usize> SerializeModule for PermutedEmbedding<A, B> {
    fn serialize(&self, s: &mut luminal::module::Serializer) {
        s.tensor("weight", self.weight.clone());
    }
}

// Single
impl<S: Dimension, const N: usize, const DIM: usize> Module<GraphTensor<(S,)>>
    for PermutedEmbedding<N, DIM>
{
    type Output = GraphTensor<(S, Const<DIM>)>;

    fn forward(&self, input: GraphTensor<(S,)>) -> Self::Output {
        self.weight.clone().permute().gather(input)
    }
}

// Batch
impl<B: Dimension, S: Dimension, const N: usize, const DIM: usize> Module<GraphTensor<(B, S)>>
    for PermutedEmbedding<N, DIM>
{
    type Output = GraphTensor<(B, S, Const<DIM>)>;

    fn forward(&self, input: GraphTensor<(B, S)>) -> Self::Output {
        self.weight
            .clone()
            .permute()
            .gather(input.dyn_reshape::<(Dyn<'-'>,), _>(&[B::size() * S::size()]))
            .reshape()
    }
}

#[cfg(test)]
mod tests {
    use dfdx::{
        prelude::Module as DfdxModule,
        tensor::{Cpu, TensorFromVec},
    };

    use luminal::prelude::Module;

    use super::Embedding;
    use dfdx::nn::BuildOnDevice;
    luminal::test_imports!();

    #[test]
    fn test_embedding() {
        let cx = Graph::new();
        let batch = cx
            .tensor::<R2<2, 3>>()
            .set(vec![1.0, 0.0, 2.0, 1.0, 0.0, 1.0]);
        let a = cx.tensor::<R1<3>>().set(vec![1.0, 0.0, 1.0]).retrieve();

        let model: Embedding<3, 4> = InitModule::initialize(&cx);
        model
            .weight
            .clone()
            .set(vec![1.1, 2., 3., 1., 2., 3., 14., 2., 33., 1., 2., 3.]);
        let mut b = model.forward(a).retrieve();
        let mut batch_out = model.forward(batch).retrieve();

        cx.compile(GenericCompiler::default(), (&mut b, &mut batch_out));

        cx.execute();

        let d_dev = Cpu::default();
        let mut d_model = <dfdx::nn::modules::builders::Embedding<3, 4>>::build_on_device(&d_dev);
        d_model.weight = d_dev.tensor_from_vec(
            vec![1.1, 2., 3., 1., 2., 3., 14., 2., 33., 1., 2., 3.],
            (DConst::<3>, DConst::<4>),
        );
        let d_a = d_dev.tensor_from_vec(vec![1, 0, 1], (DConst::<3>,));
        let d_batch = d_dev.tensor_from_vec(vec![1, 0, 2, 1, 0, 1], (DConst::<2>, DConst::<3>));

        let d_b = d_model.forward(d_a);
        let d_batch_out = d_model.forward(d_batch);

        assert_close(&b.data(), &d_b.as_vec());
        assert_close(&batch_out.data(), &d_batch_out.as_vec());
    }
}
