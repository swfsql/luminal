use crate::prelude::*;
use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::rc::Weak;

use petgraph::graph::NodeIndex;

/// A tensor on the graph.
///
/// Graphs can be built by performing operations on these tensors.
/// ```rust
/// use luminal::prelude::*;
/// let mut cx = Graph::new();
/// let a: GraphTensor<R1<3>> = cx.tensor();
/// let b: GraphTensor<R1<3>> = cx.tensor();
/// let c: GraphTensor<R1<3>> = a + b;
/// // The graph `cx` now has `a` and `b` loading nodes, and an add node resulting in `c`
/// ```
#[derive(Clone)]
pub struct GraphTensor<S: Shape> {
    pub id: NodeIndex,
    pub graph_ref: Weak<RefCell<Graph>>,
    pub(crate) _phantom: PhantomData<S>,
    pub shape: ShapeTracker,
}

impl<S: Shape> GraphTensor<S> {
    /// Create a GraphTensor from a NodeIndex
    pub fn from_id(id: NodeIndex, shape: ShapeTracker, graph_ref: Weak<RefCell<Graph>>) -> Self {
        Self {
            id,
            graph_ref,
            shape,
            _phantom: Default::default(),
        }
    }

    /// Mark this tensor to not be deleted
    pub fn keep(&mut self) -> &mut Self {
        self.graph().unwrap().keep_tensors(self.id);
        self
    }

    /// Mark this tensor to be retrieved later
    pub fn retrieve(&mut self) -> &mut Self {
        self.keep();
        self.graph()
            .unwrap()
            .borrow_mut()
            .to_retrieve
            .insert(self.id, (0, self.shape));
        self
    }

    /// Remove this tensor's data from the graph.
    pub fn drop(&self) {
        self.graph().unwrap().drop_tensors(self.id);
    }

    /// Get a mutable reference to the graph this tensor belongs to
    #[allow(clippy::mut_from_ref)]
    pub fn graph(&self) -> Option<GraphWrapper> {
        self.graph_ref.upgrade().map(GraphWrapper)
    }

    /// Set the value of the tensor, with dynamic dimensions.
    /// ```rust
    /// use luminal::prelude::*;
    /// let mut cx = Graph::new();
    /// let a: GraphTensor<(Const<2>, Dyn<'s'>)> = cx
    ///     .tensor()
    ///     .set_dyn(vec![1., 2., 3., 4.], &[2, 2]);
    /// ```
    pub fn set_dyn<T: Data + Clone>(self, data: T, shape: &[usize]) -> Self {
        // Report dyn dim values to graph dyn map
        assert_eq!(
            S::NUM_DIMS,
            shape.len(),
            "Number of dimensions do not match!"
        );
        let graph = self.graph().unwrap();
        let mut graph_mut = graph.borrow_mut();
        let mut dyn_map_mut = graph_mut.dyn_map.as_ref().borrow_mut();
        for (d, s) in S::realized_shape().iter().zip(shape.iter()) {
            if let Some(c) = d.to_symbols().pop() {
                dyn_map_mut.insert(c, *s);
            }
        }
        drop(dyn_map_mut);
        graph_mut.get_op_mut::<Function>(self.id).1 =
            Box::new(move |_| vec![Tensor::new(data.to_owned())]);
        self
    }

    /// Set the name of a tensor
    pub fn set_name(&self, name: &str) {
        self.graph()
            .unwrap()
            .borrow_mut()
            .get_op_mut::<Function>(self.id)
            .0 = name.to_string();
    }

    /// Convert tensor to a shapeless tensor
    pub fn no_shape(self) -> GraphTensor<()> {
        GraphTensor::from_id(self.id, self.shape, self.graph_ref)
    }

    /// Get the contiguous data of the tensor
    pub fn data(&self) -> Vec<f32> {
        let graph = self.graph().unwrap();
        let tensor = graph.get_tensor_ref(self.id, 0).unwrap();
        let orig_data = tensor.downcast_ref::<Vec<f32>>().unwrap();
        let mut st = self.shape;
        if !st.is_reshaped() {
            return orig_data.clone();
        }
        st.resolve_global_dyn_dims(&self.graph().unwrap().borrow().dyn_map.as_ref().borrow());
        let mut data = vec![0.; st.n_elements().to_usize().unwrap()];
        let (ind, val) = (
            st.index_expression_no_simplify(),
            st.valid_expression_no_simplify(),
        );
        #[allow(unused_mut)]
        for (i, mut r) in data.iter_mut().enumerate() {
            if val.exec_single_var(i) != 0 {
                *r = orig_data[ind.exec_single_var(i)];
            }
        }
        data
    }
}

impl<S: ConstShape> GraphTensor<S> {
    /// Set the value of the tensor matching the constant shape
    pub fn set<T: Data + Clone, D: ToData<S, T>>(self, data: D) -> Self {
        let data = data.to_data_vec();
        self.graph()
            .unwrap()
            .borrow_mut()
            .get_op_mut::<Function>(self.id)
            .1 = Box::new(move |_| vec![Tensor::new(data.to_owned())]);
        self
    }

    /// Set the tensor with a generating closure to be ran at runtime
    pub fn set_deferred(self, loader: impl Fn() -> Vec<f32> + 'static) -> Self {
        self.graph()
            .unwrap()
            .borrow_mut()
            .get_op_mut::<Function>(self.id)
            .1 = Box::new(move |_| vec![Tensor::new(loader())]);
        self
    }
}

fn pretty_print_tensor_recursive(
    f: &mut std::fmt::Formatter<'_>,
    data: &[f32],
    shape: &[usize],
    level: usize,
) -> std::fmt::Result {
    if shape.is_empty() {
        // Base case: no dimensions left
        return Ok(());
    }

    let indent = "  ".repeat(level);

    if shape.len() == 1 {
        // If this is the innermost dimension, print the raw data in a single line
        write!(f, "{}[", indent)?;
        if data.len() > 10 {
            for (i, value) in data.iter().take(5).enumerate() {
                write!(f, "{:.6}", value)?;
                if i < data.len() - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "..., ")?;
            for (i, value) in data.iter().skip(data.len() - 5).enumerate() {
                write!(f, "{:.6}", value)?;
                if i < data.len() - 1 {
                    write!(f, ", ")?;
                }
            }
        } else {
            for (i, value) in data.iter().enumerate() {
                write!(f, "{:.6}", value)?;
                if i < data.len() - 1 {
                    write!(f, ", ")?;
                }
            }
        }
        write!(f, "]")?; // No newline after the innermost array
    } else {
        // For higher dimensions, handle the nesting
        writeln!(f, "{indent}[")?;
        let stride = shape[1..].iter().product();
        if data.len() / stride > 10 {
            for (i, chunk) in data.chunks(stride).take(5).enumerate() {
                pretty_print_tensor_recursive(f, chunk, &shape[1..], level + 1)?;
                if i < shape[0] - 1 {
                    writeln!(f, ",")?; // Place the comma right after the bracket and then a newline
                }
            }
            writeln!(f, "{indent}  ..., ")?;
            for (i, chunk) in data
                .chunks(stride)
                .skip(data.len() / stride - 5)
                .enumerate()
            {
                pretty_print_tensor_recursive(f, chunk, &shape[1..], level + 1)?;
                if i < shape[0] - 1 {
                    writeln!(f, ",")?; // Place the comma right after the bracket and then a newline
                }
            }
        } else {
            for (i, chunk) in data.chunks(stride).enumerate() {
                pretty_print_tensor_recursive(f, chunk, &shape[1..], level + 1)?;
                if i < shape[0] - 1 {
                    writeln!(f, ",")?; // Place the comma right after the bracket and then a newline
                }
            }
        }
        writeln!(f)?; // Add a newline before closing the current dimension bracket
        write!(f, "{indent}]")?; // Close the current dimension bracket
    }

    // Only add a newline after the top-level closing bracket
    if level == 0 {
        writeln!(f)?;
    }

    Ok(())
}

impl<S: Shape> Debug for GraphTensor<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Print the shape
        let mut shape = self.shape;
        shape.resolve_global_dyn_dims(&self.graph().unwrap().borrow().dyn_map.as_ref().borrow());
        let shape = shape.shape_usize();
        writeln!(f, "Tensor with Shape: {:?}", shape)?;

        // Print the data by going dimension by dimension, recursively
        pretty_print_tensor_recursive(f, &self.data(), &shape, 0)
    }
}

pub trait MarkTensors {
    /// Mark all tensors in this collection to be kept
    fn keep(&self) -> Self;
    /// Mark all tensors in this collection to be retrieved
    fn retrieve(&self) -> Self;
    /// Drop all tensors in this collection
    fn drop(&self);
    /// Set data
    fn set_dyn<T: Data + Clone>(&self, data: T, shape: &[usize]) -> Self;
}

impl<S: Shape> MarkTensors for GraphTensor<S> {
    fn keep(&self) -> Self {
        GraphTensor::keep(&mut self.clone());
        self.clone()
    }

    fn retrieve(&self) -> Self {
        GraphTensor::retrieve(&mut self.clone()).clone()
    }
    fn drop(&self) {
        GraphTensor::drop(self);
    }
    fn set_dyn<T: Data + Clone>(&self, data: T, shape: &[usize]) -> Self {
        GraphTensor::set_dyn(self.clone(), data, shape);
        self.clone()
    }
}

impl<S: MarkTensors + Clone> MarkTensors for Vec<S> {
    fn keep(&self) -> Self {
        for t in self {
            t.keep();
        }
        self.clone()
    }

    fn retrieve(&self) -> Self {
        for t in self {
            t.retrieve();
        }
        self.clone()
    }

    fn drop(&self) {
        for t in self {
            t.drop();
        }
    }
    fn set_dyn<T: Data + Clone>(&self, data: T, shape: &[usize]) -> Self {
        for t in self {
            t.set_dyn(data.clone(), shape);
        }
        self.clone()
    }
}
impl<S: MarkTensors> MarkTensors for &[S] {
    fn keep(&self) -> Self {
        for t in *self {
            t.keep();
        }
        self
    }

    fn retrieve(&self) -> Self {
        for t in *self {
            t.retrieve();
        }
        self
    }

    fn drop(&self) {
        for t in *self {
            t.drop();
        }
    }
    fn set_dyn<T: Data + Clone>(&self, data: T, shape: &[usize]) -> Self {
        for t in *self {
            t.set_dyn(data.clone(), shape);
        }
        self
    }
}

macro_rules! tuple_impls {
    ([$($name:ident),+] , [$($idx:tt),+]) => {
        impl<
        $($name:
            MarkTensors + Clone, )+
        > MarkTensors for ($($name,)+) {
            fn keep(&self) -> Self {
                $(self.$idx.keep();)+
                self.clone()
            }
            fn retrieve(&self) -> Self {
                $(self.$idx.retrieve();)+
                self.clone()
            }
            fn drop(&self) {
                $(self.$idx.drop();)+
            }
            fn set_dyn<T: Data + Clone>(&self, data: T, shape: &[usize]) -> Self {
                $(self.$idx.set_dyn(data.clone(), shape);)+
                self.clone()
            }
        }
    };
}

tuple_impls!([M1], [0]);
tuple_impls!([M1, M2], [0, 1]);
tuple_impls!([M1, M2, M3], [0, 1, 2]);
tuple_impls!([M1, M2, M3, M4], [0, 1, 2, 3]);
tuple_impls!([M1, M2, M3, M4, M5], [0, 1, 2, 3, 4]);
tuple_impls!([M1, M2, M3, M4, M5, M6], [0, 1, 2, 3, 4, 5]);
tuple_impls!([M1, M2, M3, M4, M5, M6, M7], [0, 1, 2, 3, 4, 5, 6]);
tuple_impls!([M1, M2, M3, M4, M5, M6, M7, M8], [0, 1, 2, 3, 4, 5, 6, 7]);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
);

pub trait ToData<S: Shape, T> {
    fn to_data_vec(self) -> T;
}

impl<S: Shape> ToData<S, Vec<f32>> for Vec<f32> {
    fn to_data_vec(self) -> Vec<f32> {
        self
    }
}
impl ToData<R0, Vec<f32>> for f32 {
    fn to_data_vec(self) -> Vec<f32> {
        vec![self]
    }
}
impl<const A: usize> ToData<(Const<A>,), Vec<f32>> for [f32; A] {
    fn to_data_vec(self) -> Vec<f32> {
        self.to_vec()
    }
}
impl<const A: usize, const B: usize> ToData<(Const<A>, Const<B>), Vec<f32>> for [[f32; B]; A] {
    fn to_data_vec(self) -> Vec<f32> {
        self.into_iter().flat_map(|i| i.to_vec()).collect()
    }
}
impl<const A: usize, const B: usize, const C: usize>
    ToData<(Const<A>, Const<B>, Const<C>), Vec<f32>> for [[[f32; C]; B]; A]
{
    fn to_data_vec(self) -> Vec<f32> {
        self.into_iter()
            .flat_map(|i| i.into_iter().flat_map(|i| i.to_vec()))
            .collect()
    }
}
impl<const A: usize, const B: usize, const C: usize, const D: usize>
    ToData<(Const<A>, Const<B>, Const<C>, Const<D>), Vec<f32>> for [[[[f32; D]; C]; B]; A]
{
    fn to_data_vec(self) -> Vec<f32> {
        self.into_iter()
            .flat_map(|i| {
                i.into_iter()
                    .flat_map(|i| i.into_iter().flat_map(|i| i.to_vec()))
            })
            .collect()
    }
}
impl<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize>
    ToData<(Const<A>, Const<B>, Const<C>, Const<D>), Vec<f32>> for [[[[[f32; E]; D]; C]; B]; A]
{
    fn to_data_vec(self) -> Vec<f32> {
        self.into_iter()
            .flat_map(|i| {
                i.into_iter().flat_map(|i| {
                    i.into_iter()
                        .flat_map(|i| i.into_iter().flat_map(|i| i.to_vec()))
                })
            })
            .collect()
    }
}
