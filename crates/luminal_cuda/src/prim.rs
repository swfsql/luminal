use crate::{compile_and_load_kernel, get_buffer_from_tensor, input_dyn_dims, CudaData, CudaFloat};

use super::{get_idx_valid_exps, render_dyn_dim_inputs};
use itertools::Itertools;
use rustc_hash::FxHashMap;

use std::{
    any::{Any, TypeId},
    cell::RefCell,
    marker::PhantomData,
    rc::Rc,
    sync::Arc,
};

use cudarc::driver::{CudaDevice, CudaFunction, DeviceRepr, LaunchAsync, LaunchConfig};

use luminal::{
    op::{Function as LFunction, *},
    prelude::{petgraph::visit::EdgeRef, *},
};

/// Copy a tensor to the GPU
#[derive(Clone)]
pub struct CudaCopyToDevice<T>(Arc<CudaDevice>, PhantomData<T>);
crate::debug_type!(CudaCopyToDevice);

impl<T> CudaCopyToDevice<T> {
    pub fn new(dev: Arc<CudaDevice>) -> Self {
        CudaCopyToDevice(dev, Default::default())
    }
}

impl<T: CudaFloat> Operator for CudaCopyToDevice<T> {
    fn process(&mut self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().is::<CudaData<T>>() || inp[0].0.borrowed().is::<CudaData<u8>>() {
            // Already on device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        let cpu_data = inp[0].0.borrowed().downcast_ref::<Vec<f32>>().unwrap();
        let vec = cpu_data
            .iter()
            .copied()
            .map(T::from_f32)
            .collect::<Vec<_>>();
        vec![Tensor::new(CudaData(self.0.htod_sync_copy(&vec).unwrap()))]
    }
}

/// Copy a tensor from the GPU
#[derive(Clone)]
pub struct CudaCopyFromDevice<T>(Arc<CudaDevice>, PhantomData<T>);
crate::debug_type!(CudaCopyFromDevice);

impl<T> CudaCopyFromDevice<T> {
    pub fn new(dev: Arc<CudaDevice>) -> Self {
        CudaCopyFromDevice(dev, Default::default())
    }
}

impl<T: CudaFloat> Operator for CudaCopyFromDevice<T> {
    fn process(&mut self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().is::<Vec<f32>>() {
            // Already off device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        let buf = self
            .0
            .dtoh_sync_copy(get_buffer_from_tensor::<T>(&inp[0].0))
            .unwrap();
        vec![Tensor::new(
            buf.into_iter().map(T::to_f32).collect::<Vec<_>>(),
        )]
    }
}

/// Constant value on device
#[derive(Clone)]
pub struct CudaConstant<T> {
    pub value: ConstantValue,
    device: Arc<CudaDevice>,
    dyn_map: Rc<RefCell<FxHashMap<char, usize>>>,
    _phantom: PhantomData<T>,
}
impl<T> core::fmt::Debug for CudaConstant<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaConstant({:?})", self.value)
    }
}

impl<T> CudaConstant<T> {
    pub fn new(
        device: Arc<CudaDevice>,
        value: ConstantValue,
        dyn_map: Rc<RefCell<FxHashMap<char, usize>>>,
    ) -> Self {
        Self {
            value,
            device,
            dyn_map,
            _phantom: Default::default(),
        }
    }
}

impl<T: CudaFloat> Operator for CudaConstant<T> {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut a = unsafe { self.device.alloc::<T>(1).unwrap() };
        let value = match &self.value {
            ConstantValue::Expression(e) => {
                T::from_f32(e.exec(&self.dyn_map.borrow()).unwrap() as f32)
            }
            ConstantValue::Float(f) => T::from_f32(*f),
        };
        self.device.htod_copy_into(vec![value], &mut a).unwrap();
        vec![Tensor::new(CudaData(a))]
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "elementwise" {
            if let ConstantValue::Float(f) = self.value {
                return Some(Box::new(format!("{f:?}")));
            }
        }
        None
    }
}

#[macro_export]
macro_rules! cuda_unary_op {
    ($op: expr, $op_name: ident) => {
        #[derive(Clone)]
        pub struct $op_name<T> {
            function: CudaFunction,
            device: Arc<CudaDevice>,
            dyn_symbols: Vec<char>,
            dyn_map: Rc<RefCell<FxHashMap<char, usize>>>,
            _phantom: PhantomData<T>,
        }

        impl<T: CudaFloat> $op_name<T> {
            pub fn new(
                shape: ShapeTracker,
                device: Arc<CudaDevice>,
                dyn_map: Rc<RefCell<FxHashMap<char, usize>>>,
            ) -> Self {
                let (idx_exp, valid_exp) = get_idx_valid_exps(shape);
                let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[shape]);
                let type_name = T::type_name();
                let code = format!(
                    "
        #include \"cuda_fp16.h\"
        extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp, int numel{rendered}) {{
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < numel && {valid_exp} != 0) {{
                out[idx] = {}(inp[{idx_exp}]);
            }}
        }}", $op
                );
                Self {
                    function: compile_and_load_kernel(code, &device),
                    device,
                    dyn_symbols,
                    dyn_map,
                    _phantom: Default::default(),
                }
            }
        }

        impl<T: CudaFloat> Operator for $op_name<T> {
            fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
                let inp = get_buffer_from_tensor::<T>(&tensors[0].0);
                let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
                let out = self.device.alloc_zeros::<T>(inp_size).unwrap();
                let mut params = vec![
                    (&out).as_kernel_param(),
                    inp.as_kernel_param(),
                    inp_size.as_kernel_param(),
                ];
                input_dyn_dims(&mut params, &self.dyn_symbols, &self.dyn_map.as_ref().borrow());
                unsafe {
                    self.function
                        .clone()
                        .launch(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                        .unwrap();
                }

                vec![Tensor::new(CudaData(out))]
            }

            fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
                if key == "elementwise" {
                    return Some(Box::new(format!("{}(input0)", $op)));
                }

                None
            }
        }

        $crate::debug_type!($op_name);
    };
}

cuda_unary_op!("", CudaContiguous);
cuda_unary_op!("log2", CudaLog2);
cuda_unary_op!("exp2", CudaExp2);
cuda_unary_op!(if T::is_f32() { "sqrt" } else { "hsqrt" }, CudaSqrt);
cuda_unary_op!("sin", CudaSin);
cuda_unary_op!(if T::is_f32() { "__frcp_rn" } else { "hrcp" }, CudaRecip);

#[derive(Clone)]
pub struct CudaAdd<T> {
    function: CudaFunction,
    device: Arc<CudaDevice>,
    _phantom: PhantomData<T>,
    dyn_symbols: Vec<char>,
    dyn_map: Rc<RefCell<FxHashMap<char, usize>>>,
}
crate::debug_type!(CudaAdd);

impl<T: CudaFloat> CudaAdd<T> {
    pub fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        device: Arc<CudaDevice>,
        dyn_map: Rc<RefCell<FxHashMap<char, usize>>>,
    ) -> Self {
        let (a_idx, a_valid) = get_idx_valid_exps(a_shape);
        let (b_idx, b_valid) = get_idx_valid_exps(b_shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[a_shape, b_shape]);
        let type_name = T::type_name();
        let code = format!(
            "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp_a, const {type_name} *inp_b, int numel{rendered}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        out[idx] =
            (({a_valid}) == 0 ? ({type_name})0.0 : inp_a[{a_idx}])
            + (({b_valid}) == 0 ? ({type_name})0.0 : inp_b[{b_idx}]);
    }}
}}");
        Self {
            function: compile_and_load_kernel(code, &device),
            device,
            _phantom: Default::default(),
            dyn_symbols,
            dyn_map,
        }
    }
}

impl<T: CudaFloat> Operator for CudaAdd<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = get_buffer_from_tensor::<T>(&tensors[0].0);
        let b = get_buffer_from_tensor::<T>(&tensors[1].0);
        let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
        let out = unsafe { self.device.alloc::<T>(inp_size).unwrap() };
        let mut params = vec![
            (&out).as_kernel_param(),
            a.as_kernel_param(),
            b.as_kernel_param(),
            inp_size.as_kernel_param(),
        ];
        input_dyn_dims(
            &mut params,
            &self.dyn_symbols,
            &self.dyn_map.as_ref().borrow(),
        );

        unsafe {
            self.function
                .clone()
                .launch(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }

        vec![Tensor::new(CudaData(out))]
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "elementwise" {
            return Some(Box::new("input0 + input1".to_string()));
        }
        None
    }
}

#[derive(Clone)]
pub struct CudaMul<T> {
    function: CudaFunction,
    device: Arc<CudaDevice>,
    _phantom: PhantomData<T>,
    dyn_symbols: Vec<char>,
    dyn_map: Rc<RefCell<FxHashMap<char, usize>>>,
}
crate::debug_type!(CudaMul);

impl<T: CudaFloat> CudaMul<T> {
    pub fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        device: Arc<CudaDevice>,
        dyn_map: Rc<RefCell<FxHashMap<char, usize>>>,
    ) -> Self {
        let (a_idx, a_valid) = get_idx_valid_exps(a_shape);
        let (b_idx, b_valid) = get_idx_valid_exps(b_shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[a_shape, b_shape]);
        let type_name = T::type_name();
        let code = format!("
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp_a, const {type_name} *inp_b, int numel{rendered}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        out[idx] = (({a_valid}) == 0 ? ({type_name})0.0 : inp_a[{a_idx}]) * (({b_valid}) == 0 ? ({type_name})0.0 : inp_b[{b_idx}]);
    }}
}}");
        Self {
            function: compile_and_load_kernel(code, &device),
            device,
            _phantom: Default::default(),
            dyn_symbols,
            dyn_map,
        }
    }
}

impl<T: CudaFloat> Operator for CudaMul<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = get_buffer_from_tensor::<T>(&tensors[0].0);
        let b = get_buffer_from_tensor::<T>(&tensors[1].0);
        let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
        let out = unsafe { self.device.alloc::<T>(inp_size).unwrap() };
        let mut params = vec![
            (&out).as_kernel_param(),
            a.as_kernel_param(),
            b.as_kernel_param(),
            inp_size.as_kernel_param(),
        ];
        input_dyn_dims(
            &mut params,
            &self.dyn_symbols,
            &self.dyn_map.as_ref().borrow(),
        );

        unsafe {
            self.function
                .clone()
                .launch(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }

        vec![Tensor::new(CudaData(out))]
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "elementwise" {
            return Some(Box::new("input0 * input1".to_string()));
        }
        None
    }
}

#[derive(Clone)]
pub struct CudaMod<T> {
    function: CudaFunction,
    device: Arc<CudaDevice>,
    _phantom: PhantomData<T>,
    dyn_symbols: Vec<char>,
    dyn_map: Rc<RefCell<FxHashMap<char, usize>>>,
}
crate::debug_type!(CudaMod);

impl<T: CudaFloat> CudaMod<T> {
    pub fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        device: Arc<CudaDevice>,
        dyn_map: Rc<RefCell<FxHashMap<char, usize>>>,
    ) -> Self {
        let (a_idx, a_valid) = get_idx_valid_exps(a_shape);
        let (b_idx, b_valid) = get_idx_valid_exps(b_shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[a_shape, b_shape]);
        let type_name = T::type_name();
        let code = format!("
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp_a, const {type_name} *inp_b, int numel{rendered}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        out[idx] = fmod((({a_valid}) == 0 ? ({type_name})0.0 : inp_a[{a_idx}]), (({b_valid}) == 0 ? ({type_name})0.0 : inp_b[{b_idx}]));
    }}
}}");
        Self {
            function: compile_and_load_kernel(code, &device),
            device,
            _phantom: Default::default(),
            dyn_symbols,
            dyn_map,
        }
    }
}

impl<T: CudaFloat> Operator for CudaMod<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = get_buffer_from_tensor::<T>(&tensors[0].0);
        let b = get_buffer_from_tensor::<T>(&tensors[1].0);
        let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
        let out = unsafe { self.device.alloc::<T>(inp_size).unwrap() };
        let mut params = vec![
            (&out).as_kernel_param(),
            a.as_kernel_param(),
            b.as_kernel_param(),
            inp_size.as_kernel_param(),
        ];
        input_dyn_dims(
            &mut params,
            &self.dyn_symbols,
            &self.dyn_map.as_ref().borrow(),
        );

        unsafe {
            self.function
                .clone()
                .launch(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }

        vec![Tensor::new(CudaData(out))]
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "elementwise" {
            return Some(Box::new("fmod(input0, input1)".to_string()));
        }
        None
    }
}

#[derive(Clone)]
pub struct CudaLessThan<T> {
    function: CudaFunction,
    device: Arc<CudaDevice>,
    _phantom: PhantomData<T>,
    dyn_symbols: Vec<char>,
    dyn_map: Rc<RefCell<FxHashMap<char, usize>>>,
}
crate::debug_type!(CudaLessThan);

impl<T: CudaFloat> CudaLessThan<T> {
    pub fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        device: Arc<CudaDevice>,
        dyn_map: Rc<RefCell<FxHashMap<char, usize>>>,
    ) -> Self {
        let (a_idx, a_valid) = get_idx_valid_exps(a_shape);
        let (b_idx, b_valid) = get_idx_valid_exps(b_shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[a_shape, b_shape]);
        let type_name = T::type_name();
        let code = format!("
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp_a, const {type_name} *inp_b, int numel{rendered}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        {type_name} a_t = (({a_valid}) != 0) ? inp_a[{a_idx}] : ({type_name})0.0;
        {type_name} b_t = (({b_valid}) != 0) ? inp_b[{b_idx}] : ({type_name})0.0;
        if (a_t < b_t) {{
            out[idx] = ({type_name})1.0;
        }} else {{
            out[idx] = ({type_name})0.0;
        }}
    }}
}}");
        Self {
            function: compile_and_load_kernel(code, &device),
            device,
            _phantom: Default::default(),
            dyn_symbols,
            dyn_map,
        }
    }
}

impl<T: CudaFloat> Operator for CudaLessThan<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = get_buffer_from_tensor::<T>(&tensors[0].0);
        let b = get_buffer_from_tensor::<T>(&tensors[1].0);
        let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
        let out = unsafe { self.device.alloc::<T>(inp_size).unwrap() };
        let mut params = vec![
            (&out).as_kernel_param(),
            a.as_kernel_param(),
            b.as_kernel_param(),
            inp_size.as_kernel_param(),
        ];
        input_dyn_dims(
            &mut params,
            &self.dyn_symbols,
            &self.dyn_map.as_ref().borrow(),
        );

        unsafe {
            self.function
                .clone()
                .launch(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }

        vec![Tensor::new(CudaData(out))]
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "elementwise" {
            return Some(Box::new("(float)(input0 < input1 ? 1.0 : 0.0)".to_string()));
        }
        None
    }
}

#[derive(Clone)]
pub struct CudaSumReduce<T> {
    function: CudaFunction,
    pub device: Arc<CudaDevice>,
    pub dim: usize,
    _phantom: PhantomData<T>,
    dyn_symbols: Vec<char>,
    dyn_map: Rc<RefCell<FxHashMap<char, usize>>>,
}
crate::debug_type!(CudaSumReduce);

impl<T: CudaFloat> CudaSumReduce<T> {
    pub fn new(
        dim: usize,
        shape: ShapeTracker,
        device: Arc<CudaDevice>,
        dyn_map: Rc<RefCell<FxHashMap<char, usize>>>,
    ) -> Self {
        let (idx, valid) = get_idx_valid_exps(shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[shape]);
        let type_name = T::type_name();
        let code = format!("#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp, const int front_size, const int back_size, const int dim_size, int numel{rendered}) {{
    int i_ = blockIdx.x * blockDim.x + threadIdx.x;

    if (i_ < numel) {{
        int a_ = i_ / back_size;
        int b_ = i_ % back_size;
        float reduce_value = 0.0;
        for (int c_ = 0; c_ < dim_size; c_++) {{
            int idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if (({valid}) != 0) {{
                reduce_value = reduce_value + (float)inp[{idx}];
            }}
        }}
        out[i_] = ({type_name})reduce_value;
    }}
}}");
        Self {
            function: compile_and_load_kernel(code, &device),
            device,
            dim,
            _phantom: Default::default(),
            dyn_symbols,
            dyn_map,
        }
    }
}
impl<T> Operator for CudaSumReduce<T>
where
    T: CudaFloat,
    CudaData<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut shape = tensors[0].1;
        shape.remove_dim(self.dim);
        let inp_size = shape.n_elements().to_usize().unwrap();
        let inp = get_buffer_from_tensor::<T>(&tensors[0].0);
        let front_size: usize = tensors[0]
            .1
            .shape()
            .iter()
            .take(self.dim)
            .map(|i| i.to_usize().unwrap())
            .product();
        let back_size: usize = tensors[0]
            .1
            .shape()
            .iter()
            .skip(self.dim + 1)
            .map(|i| i.to_usize().unwrap())
            .product();
        let dim_size = tensors[0].1.shape()[self.dim].to_usize().unwrap();

        let out = self.device.alloc_zeros::<T>(inp_size).unwrap();
        let mut params = vec![
            (&out).as_kernel_param(),
            inp.as_kernel_param(),
            front_size.as_kernel_param(),
            back_size.as_kernel_param(),
            dim_size.as_kernel_param(),
            inp_size.as_kernel_param(),
        ];
        input_dyn_dims(
            &mut params,
            &self.dyn_symbols,
            &self.dyn_map.as_ref().borrow(),
        );
        unsafe {
            self.function
                .clone()
                .launch(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }
        vec![Tensor::new(CudaData(out))]
    }
}

#[derive(Clone)]
pub struct CudaMaxReduce<T> {
    function: CudaFunction,
    pub device: Arc<CudaDevice>,
    pub dim: usize,
    _phantom: PhantomData<T>,
    dyn_symbols: Vec<char>,
    dyn_map: Rc<RefCell<FxHashMap<char, usize>>>,
}
crate::debug_type!(CudaMaxReduce);

impl<T: CudaFloat> CudaMaxReduce<T> {
    pub fn new(
        dim: usize,
        shape: ShapeTracker,
        device: Arc<CudaDevice>,
        dyn_map: Rc<RefCell<FxHashMap<char, usize>>>,
    ) -> Self {
        let (idx, valid) = get_idx_valid_exps(shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[shape]);
        let type_name = T::type_name();
        let code = format!("#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp, const int front_size, const int back_size, const int dim_size, int numel{rendered}) {{
    int i_ = blockIdx.x * blockDim.x + threadIdx.x;

    if (i_ < numel) {{
        int a_ = i_ / back_size;
        int b_ = i_ % back_size;
        float reduce_value = -__int_as_float(0x7f800000);
        for (int c_ = 0; c_ < dim_size; c_++) {{
            int idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if (({valid}) != 0) {{
                reduce_value = max(reduce_value, (float)inp[{idx}]);
            }}
        }}
        out[i_] = ({type_name})reduce_value;
    }}
}}");
        Self {
            function: compile_and_load_kernel(code, &device),
            device,
            dim,
            _phantom: Default::default(),
            dyn_symbols,
            dyn_map,
        }
    }
}
impl<T: CudaFloat> Operator for CudaMaxReduce<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut shape = tensors[0].1;
        shape.remove_dim(self.dim);
        let inp_size = shape.n_elements().to_usize().unwrap();
        let inp = get_buffer_from_tensor::<T>(&tensors[0].0);
        let front_size: usize = tensors[0]
            .1
            .shape()
            .iter()
            .take(self.dim)
            .map(|i| i.to_usize().unwrap())
            .product();
        let back_size: usize = tensors[0]
            .1
            .shape()
            .iter()
            .skip(self.dim + 1)
            .map(|i| i.to_usize().unwrap())
            .product();
        let dim_size = tensors[0].1.shape()[self.dim].to_usize().unwrap();

        let out = self.device.alloc_zeros::<T>(inp_size).unwrap();
        let mut params = vec![
            (&out).as_kernel_param(),
            inp.as_kernel_param(),
            front_size.as_kernel_param(),
            back_size.as_kernel_param(),
            dim_size.as_kernel_param(),
            inp_size.as_kernel_param(),
        ];
        input_dyn_dims(
            &mut params,
            &self.dyn_symbols,
            &self.dyn_map.as_ref().borrow(),
        );
        unsafe {
            self.function
                .clone()
                .launch(LaunchConfig::for_num_elems(inp_size as u32), &mut params)
                .unwrap();
        }
        vec![Tensor::new(CudaData(out))]
    }
}

/// Convert all primitive ops to cuda primitive ops, and insert copy to and from device ops
#[derive(Debug, Default)]
pub struct PrimitiveCompiler<T>(PhantomData<T>);

impl<T: CudaFloat> Compiler for PrimitiveCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &GraphWrapper, mut ids: To) {
        let dev = CudaDevice::new(0).unwrap();
        // Go through the graph and insert copy ops
        // Copy function output to device and input from device
        let graph_ref = graph.borrow();
        let function_nodes = graph_ref
            .node_indices()
            .filter(|n| {
                graph_ref.node_weight(*n).unwrap().as_any().is::<Function>()
                    && graph_ref.edges(*n).count() != 0
            })
            .collect::<Vec<_>>();
        drop(graph_ref);
        for function_node in function_nodes {
            // Create copy node
            let copy_node = graph
                .add_op(CudaCopyToDevice::<T>::new(dev.clone()))
                .input(function_node, 0, ShapeTracker::new(&[]))
                .finish();

            let mut graph_mut = graph.borrow_mut();

            // Switch outgoing edges from input to copy_node
            for (edge_id, weight, dest) in graph_mut
                .edges_directed(function_node, petgraph::Direction::Outgoing)
                .map(|e| (e.id(), *e.weight(), e.target()))
                .filter(|(_, _, trg)| *trg != copy_node)
                .collect::<Vec<_>>()
            {
                graph_mut.add_edge(copy_node, dest, weight);
                graph_mut.remove_edge(edge_id);
            }

            if graph_mut.no_delete.remove(&function_node) {
                graph_mut.no_delete.insert(copy_node);
            }
            drop(graph_mut);
            let v = graph.borrow().to_retrieve.get(&function_node).cloned();
            let mut graph_mut = graph.borrow_mut();
            if let Some(v) = v {
                graph_mut.to_retrieve.insert(copy_node, v);
            }
            drop(graph_mut);

            // Insert copy from device for function inputs
            let elems = graph
                .borrow()
                .edges_directed(function_node, petgraph::Direction::Incoming)
                .map(|e| (e.source(), e.id(), *e.weight()))
                .collect::<Vec<_>>();
            for (source, edge, edge_weight) in elems {
                let copy_from_node = graph
                    .add_op(CudaCopyFromDevice::<T>::new(dev.clone()))
                    .input(source, 0, ShapeTracker::new(&[]))
                    .finish();
                let mut graph_mut = graph.borrow_mut();
                graph_mut.add_edge(copy_from_node, function_node, edge_weight);
                graph_mut.remove_edge(edge);
            }
        }

        let graph_ref = graph.borrow();

        // Copy to_retrieve from device
        let elems = graph_ref
            .to_retrieve
            .iter()
            .map(|(a, b)| (*a, *b))
            // Filter to non-functions
            .filter(|(n, _)| {
                !graph_ref
                    .node_weight(*n)
                    .unwrap()
                    .as_any()
                    .is::<LFunction>()
            })
            .collect::<Vec<_>>();
        drop(graph_ref);
        for (output_node, (_, output_shape)) in elems {
            if graph
                .borrow()
                .node_weight(output_node)
                .unwrap()
                .as_any()
                .is::<CudaCopyToDevice<T>>()
            {
                // This output is already a copy to, instead of adding a copy from, let's remap back to the source
                let src = graph
                    .borrow()
                    .neighbors_directed(output_node, petgraph::Direction::Incoming)
                    .next()
                    .unwrap();
                let mut graph_mut = graph.borrow_mut();
                graph_mut.no_delete.remove(&output_node);
                graph_mut.no_delete.insert(src);
                let w = graph_mut.to_retrieve.remove(&output_node).unwrap();
                graph_mut.to_retrieve.insert(src, w);
            } else {
                // Create copy node
                let copy_node = graph
                    .add_op(CudaCopyFromDevice::<T>::new(dev.clone()))
                    .input(output_node, 0, output_shape)
                    .finish();

                remap(output_node, copy_node, &mut ids, &mut graph.borrow_mut());
            }
        }

        fn is<T: Any>(type_id: TypeId) -> bool {
            type_id == TypeId::of::<T>()
        }

        let nodes = graph.borrow().node_indices().collect::<Vec<_>>();

        // Swap primitive ops
        for id in nodes {
            let mut graph_mut = graph.borrow_mut();
            let shapes = graph_mut
                .edges_directed(id, petgraph::Direction::Incoming)
                .filter_map(|i| i.weight().as_data())
                .sorted_by_key(|e| e.0)
                .map(|e| e.2)
                .collect::<Vec<_>>();
            let op = graph_mut.node_weight(id).unwrap().as_any().type_id();
            let dyn_map = graph_mut.dyn_map.clone();
            let op_ref = graph_mut.graph.node_weight_mut(id).unwrap();
            if is::<Log2>(op) {
                *op_ref = Box::new(CudaLog2::<T>::new(shapes[0], dev.clone(), dyn_map));
            } else if is::<Exp2>(op) {
                *op_ref = Box::new(CudaExp2::<T>::new(shapes[0], dev.clone(), dyn_map));
            } else if is::<Sin>(op) {
                *op_ref = Box::new(CudaSin::<T>::new(shapes[0], dev.clone(), dyn_map));
            } else if let Some(c) = op_ref.as_any().downcast_ref::<Constant>() {
                *op_ref = Box::new(CudaConstant::<T>::new(dev.clone(), c.0.clone(), dyn_map));
            } else if is::<Recip>(op) {
                *op_ref = Box::new(CudaRecip::<T>::new(shapes[0], dev.clone(), dyn_map));
            } else if is::<Sqrt>(op) {
                *op_ref = Box::new(CudaSqrt::<T>::new(shapes[0], dev.clone(), dyn_map));
            } else if is::<Add>(op) {
                *op_ref = Box::new(CudaAdd::<T>::new(
                    shapes[0],
                    shapes[1],
                    dev.clone(),
                    dyn_map,
                ));
            } else if is::<Mul>(op) {
                *op_ref = Box::new(CudaMul::<T>::new(
                    shapes[0],
                    shapes[1],
                    dev.clone(),
                    dyn_map,
                ));
            } else if is::<Mod>(op) {
                *op_ref = Box::new(CudaMod::<T>::new(
                    shapes[0],
                    shapes[1],
                    dev.clone(),
                    dyn_map,
                ));
            } else if is::<LessThan>(op) {
                *op_ref = Box::new(CudaLessThan::<T>::new(
                    shapes[0],
                    shapes[1],
                    dev.clone(),
                    dyn_map,
                ));
            } else if is::<Contiguous>(op) {
                *op_ref = Box::new(CudaContiguous::<T>::new(shapes[0], dev.clone(), dyn_map));
            } else if let Some(SumReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(CudaSumReduce::<T>::new(
                    *dim,
                    shapes[0],
                    dev.clone(),
                    dyn_map,
                ));
            } else if let Some(MaxReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(CudaMaxReduce::<T>::new(
                    *dim,
                    shapes[0],
                    dev.clone(),
                    dyn_map,
                ));
            }
        }
    }
}
