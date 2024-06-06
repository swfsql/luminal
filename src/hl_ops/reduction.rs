use itertools::Itertools;

use crate::{
    op::{self},
    prelude::*,
};

impl<S: Shape> GraphTensor<S> {
    pub fn sum_reduce<Dst: Shape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        let mut shape = self.shape;

        let mut new_id = self.id;
        for dim in Ax::as_array().into_iter().collect_vec().into_iter().rev() {
            new_id = self
                .graph()
                .unwrap()
                .add_op(op::SumReduce(dim))
                .input(new_id, 0, shape)
                .finish();
            // Reduce shape
            shape.remove_dim(dim);
        }
        GraphTensor::from_id(new_id, shape, self.graph_ref)
    }

    pub fn max_reduce<Dst: Shape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        let mut shape = self.shape;

        let mut new_id = self.id;
        for dim in Ax::as_array().into_iter().collect_vec().into_iter().rev() {
            new_id = self
                .graph()
                .unwrap()
                .add_op(op::MaxReduce(dim))
                .input(new_id, 0, shape)
                .finish();
            // Reduce shape
            shape.remove_dim(dim);
        }
        GraphTensor::from_id(new_id, shape, self.graph_ref)
    }

    pub fn mean_reduce<Dst: Shape, Ax: Axes>(self) -> GraphTensor<Dst>
    where
        S: HasAxes<Ax> + ReduceShapeTo<Dst, Ax>,
    {
        let mut shape = self.shape;
        let mut node_id = self.id;
        for dim in Ax::as_array().into_iter().collect_vec().into_iter().rev() {
            // Sum reduce
            node_id = self
                .graph()
                .unwrap()
                .add_op(op::SumReduce(dim))
                .input(node_id, 0, shape)
                .finish();

            // Divide by size of dimension
            let div_tensor = self
                .graph()
                .unwrap()
                .constant_expr(shape.remove_dim(dim))
                .id;
            let mul_tensor = self
                .graph()
                .unwrap()
                .add_op(op::Recip)
                .input(div_tensor, 0, ShapeTracker::new(&[]))
                .finish();
            node_id = self
                .graph()
                .unwrap()
                .add_op(op::Mul)
                .input(node_id, 0, shape)
                .input(
                    mul_tensor,
                    0,
                    ShapeTracker::fake(
                        &shape
                            .shape()
                            .iter()
                            .map(Expression::from)
                            .collect::<Vec<_>>(),
                    ),
                )
                .finish();
        }
        GraphTensor::from_id(node_id, shape, self.graph_ref)
    }
}

#[cfg(test)]
mod tests {
    crate::test_imports!();

    #[test]
    fn test_sum_reduce() {
        let cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor::<R2<2, 3>>();
        let a = a.set(a_data.clone());
        let b = a.sum_reduce::<_, LAxis<1>>().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.sum::<_, DAxis<1>>();

        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_max_reduce() {
        let cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor::<R2<2, 3>>();
        let a = a.set(a_data.clone());
        let b = a.max_reduce::<_, LAxis<1>>().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.max::<_, DAxis<1>>();

        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_mean_reduce() {
        let cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor::<R2<2, 3>>();
        let a = a.set(a_data.clone());
        let b = a.mean_reduce::<_, LAxis<1>>().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.mean::<_, DAxis<1>>();

        assert_close(&b.data(), &d_b.as_vec());
    }
}
