use crate::{op, prelude::*};
use std::ops::{Add, Mul, Neg};

impl<S: Shape> Neg for GraphTensor<S> {
    type Output = GraphTensor<S>;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl<S: Shape> GraphTensor<S> {
    /// Base 2 log
    pub fn log2(self) -> GraphTensor<S> {
        let new_id = self
            .graph()
            .unwrap()
            .add_op(op::Log2)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    /// Base 2 exp
    pub fn exp2(self) -> GraphTensor<S> {
        let new_id = self
            .graph()
            .unwrap()
            .add_op(op::Exp2)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    /// Natural exp
    pub fn exp(self) -> GraphTensor<S> {
        (self * (1.0 / f32::ln(2.))).exp2()
    }

    /// Natural log
    pub fn ln(self) -> GraphTensor<S> {
        self.log2() * f32::ln(2.)
    }

    /// Take the reciprocal of each element
    pub fn recip(self) -> GraphTensor<S> {
        let new_id = self
            .graph()
            .unwrap()
            .add_op(op::Recip)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    /// The sin(x) function
    pub fn sin(self) -> GraphTensor<S> {
        let new_id = self
            .graph()
            .unwrap()
            .add_op(op::Sin)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    /// The cos(x) function
    pub fn cos(self) -> GraphTensor<S> {
        ((std::f32::consts::PI / 2.) - self).sin()
    }

    /// Square every element in the tensor
    pub fn square(self) -> GraphTensor<S> {
        self.clone() * self
    }

    /// The square root function
    pub fn sqrt(self) -> GraphTensor<S> {
        let new_id = self
            .graph()
            .unwrap()
            .add_op(op::Sqrt)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    /// Scale so std is 1.0
    pub fn std_norm<Ax: Axes, T>(self, epsilon: T) -> GraphTensor<S>
    where
        <S as ReduceShape<Ax>>::Reduced: Shape,
        GraphTensor<<S as ReduceShape<Ax>>::Reduced>:
            Add<T, Output = GraphTensor<<S as ReduceShape<Ax>>::Reduced>>,
        S: ReduceShape<Ax>,
    {
        (self.clone() * self.clone())
            .mean_reduce::<<S as ReduceShape<Ax>>::Reduced, _>()
            .add(epsilon)
            .sqrt()
            .recip()
            .expand_to(self.shape)
            .mul(self)
    }

    /// Center so mean is 0.0
    pub fn mean_norm<Ax: Axes>(self) -> GraphTensor<S>
    where
        <S as ReduceShape<Ax>>::Reduced: Shape,
        S: ReduceShape<Ax>,
    {
        self.clone()
            - self
                .clone()
                .mean_reduce::<<S as ReduceShape<Ax>>::Reduced, _>()
                .expand_to(self.shape)
    }

    /// Applies a layer norm along an axis
    pub fn layer_norm<Ax: Axes, T>(self, epsilon: T) -> GraphTensor<S>
    where
        <S as ReduceShape<Ax>>::Reduced: Shape,
        GraphTensor<<S as ReduceShape<Ax>>::Reduced>:
            Add<T, Output = GraphTensor<<S as ReduceShape<Ax>>::Reduced>>,
        S: ReduceShape<Ax>,
    {
        self.mean_norm::<Ax>().std_norm::<Ax, T>(epsilon)
    }

    /// Applies a softmax function along an axis
    pub fn softmax<Ax: Axes>(self) -> GraphTensor<S>
    where
        <S as ReduceShape<Ax>>::Reduced: Shape,
        S: ReduceShape<Ax>,
    {
        let m = self.clone()
            - self
                .clone()
                .max_reduce::<<S as ReduceShape<Ax>>::Reduced, _>()
                .expand_to(self.shape);
        let exp = m.exp();
        exp.clone()
            / exp
                .sum_reduce::<<S as ReduceShape<Ax>>::Reduced, _>()
                .expand()
    }

    /// Applies a log softmax function along an axis
    pub fn log_softmax<Ax: Axes>(self) -> GraphTensor<S>
    where
        <S as ReduceShape<Ax>>::Reduced: Shape,
        S: ReduceShape<Ax>,
    {
        let m = self.clone()
            - self
                .clone()
                .max_reduce::<<S as ReduceShape<Ax>>::Reduced, _>()
                .expand_to(self.shape);
        m.clone()
            - m.exp()
                .sum_reduce::<<S as ReduceShape<Ax>>::Reduced, _>()
                .ln()
                .expand()
    }

    /// Get the indicies of the max elements along the last axis
    pub fn argmax(self) -> GraphTensor<<S as ReduceShape<<S as Shape>::LastAxis>>::Reduced> {
        // Get one-hot along last dimension
        let x_equal = self.clone().equals(
            self.clone()
                .max_reduce::<_, S::LastAxis>()
                .expand_to(self.shape),
        );
        // Create index arange for last dimension
        let r = self
            .graph()
            .unwrap()
            .constant(1.)
            .expand_to::<(Dyn<'-'>,)>(ShapeTracker::new(&[self
                .shape
                .shape()
                .last()
                .unwrap()
                .small()]))
            .cumsum_last_dim()
            - 1.;
        // Multiply one-hot by expanded index arange
        (x_equal * r.expand_to(self.shape)).max_reduce()
    }

    /// Take the absolute value
    pub fn abs(self) -> GraphTensor<S> {
        self.clone().relu() + (-self).relu()
    }

    /// Get the sign of each element, '1' for positive and '-1' for negative
    pub fn sign(self) -> GraphTensor<S> {
        self.clone() / (self.abs() + 1e-10)
    }

    /// The Rectified Linear Unit activation function
    pub fn relu(self) -> GraphTensor<S> {
        self.max_f32(0.)
    }

    /// The sigmoid activation function
    pub fn sigmoid(self) -> GraphTensor<S> {
        // Based on https://github.com/tinygrad/tinygrad/blob/9d142430cbe61121c864c0015f1de83c94a7d2c0/tinygrad/mlops.py#L70
        1. / (1. + (-self).exp())
    }

    /// The swish activation function
    pub fn swish(self) -> GraphTensor<S> {
        self.clone() * self.sigmoid()
    }

    /// The tanh activation function
    pub fn tanh(self) -> GraphTensor<S> {
        (self * 2.0).sigmoid() * 2.0 - 1.0
    }

    /// The leaky relu activation function
    pub fn leaky_relu(self, neg_slope: f32) -> GraphTensor<S> {
        self.clone().relu() - (self * -neg_slope).relu()
    }

    /// The Gaussian Error Linear Unit activation function
    #[allow(clippy::excessive_precision)]
    pub fn gelu(self) -> GraphTensor<S> {
        // Based on https://github.com/tinygrad/tinygrad/blob/9fc4465557831b614b56dd645eebc940ca0fa1bb/tinygrad/tensor.py#L1162C26-L1162C104
        0.5 * self.clone()
            * (1. + (0.7978845608 * self.clone() * (1. + 0.044715 * self.clone() * self)).tanh())
    }
}

#[cfg(test)]
mod tests {
    crate::test_imports!();

    #[test]
    fn test_exp() {
        let cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor::<R2<2, 3>>().set(a_data.clone());
        let b = a.exp().retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.exp();

        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_layer_norm() {
        let cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor::<R2<2, 3>>().set(a_data.clone());
        let b = a.clone().layer_norm::<LAxis<0>, _>(1e-5).retrieve();
        let c = a.layer_norm::<LAxis<1>, _>(1e-5).retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.clone().normalize::<DAxis<0>>(1e-5);
        let d_c = d_a.normalize::<DAxis<1>>(1e-5);

        assert_close(&b.data(), &d_b.as_vec());
        assert_close(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_softmax() {
        let cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor::<R2<2, 3>>().set(a_data.clone());
        let b = a.softmax::<LAxis<1>>().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.softmax::<DAxis<1>>();

        let r = b.data();
        assert_close(&r, &d_b.as_vec());
    }

    #[test]
    fn test_sin() {
        let cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor::<R2<2, 3>>().set(a_data.clone());
        let b = a.sin().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.sin();

        let r = b.data();
        assert_close(&r, &d_b.as_vec());
    }

    #[test]
    fn test_cos() {
        let cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor::<R2<2, 3>>().set(a_data.clone());
        let b = a.cos().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.cos();
        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_relu() {
        let cx = Graph::new();
        let a_data = random_vec(4);
        let a = cx
            .tensor::<(Dyn<'a'>, Dyn<'b'>)>()
            .set_dyn(a_data.clone(), &[2, 2]);
        let b = a.relu().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.relu();
        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_gelu() {
        let cx = Graph::new();
        let a_data = random_vec(4);
        let a = cx
            .tensor::<(Dyn<'a'>, Dyn<'b'>)>()
            .set_dyn(a_data.clone(), &[2, 2]);
        let b = a.gelu().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.fast_gelu();
        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_sigmoid() {
        let cx = Graph::new();
        let a_data = random_vec(4);
        let a = cx
            .tensor::<(Dyn<'a'>, Dyn<'b'>)>()
            .set_dyn(a_data.clone(), &[2, 2]);
        let b = a.sigmoid().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.sigmoid();
        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_swish() {
        let cx = Graph::new();
        let a_data = random_vec(4);
        let a = cx
            .tensor::<(Dyn<'a'>, Dyn<'b'>)>()
            .set_dyn(a_data.clone(), &[2, 2]);
        let b = a.swish().retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.clone() * d_a.sigmoid();
        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_tanh() {
        let cx = Graph::new();
        let a_data = random_vec(4);
        let a = cx
            .tensor::<(Dyn<'a'>, Dyn<'b'>)>()
            .set_dyn(a_data.clone(), &[2, 2]);
        let b = a.tanh().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.tanh();
        assert_close(&b.data(), &d_b.as_vec());
    }
}
