# Operations

## Tensor operations

Most basic tensor operations are represented in Oolong.

Tensors of the same shape can be added together.
```jldoctest
using TensorDSL
R2 = RnSpace(2)
A = Tensor([1 2; 3 4], R2, R2)
B = VariableTensor("B", R2, R2)
A + B
# output
AddOperation{2}
    Tensor{Int64,2}, [1 2; 3 4], shape: R^2⊗ R^2
    VariableTensor{2}, B, shape: R^2⊗ R^2
```

Outer product
```jldoctest
using TensorDSL
R2 = RnSpace(2)
A = Tensor([1 2; 3 4], R2, R2)
C = ZeroTensor(R2)
A ⊗ C

# output
OuterProductOperation{3}
    Tensor{Int64,2}, [1 2; 3 4], shape: R^2⊗ R^2
    ZeroTensor{1}, shape: R^2
```

Tensors can be indexed using both [`FreeIndex`](@ref) objects and integers.
```jldoctest
using TensorDSL
V = VectorSpace()
x = FreeIndex(V, "x")
A = VariableTensor("A", VectorSpace(3), V)
A[1, x]
# output
IndexingOperation{0}
    VariableTensor{2}, A, shape: V3_2⊗ V_1
    (1, x)
```
Component tensor is an operation where a free index of a tensor expression is
"rolled out". This turn a free index into a shape in the tensor expression.
```jldoctest
using TensorDSL
V = VectorSpace()
x = FreeIndex(V, "x")
A = VariableTensor("A", V', VectorSpace(3))
componenttensor(A[x', 1], x')
# output
ComponentTensorOperation{1}
    IndexingOperation{0}
        VariableTensor{2}, A, shape: V_3*⊗ V3_4
        (x*, 1)
    (x*,)
```
Tensor contraction is invoked whenever the expression is indexed with an upper
and lower index.
```jldoctest
using TensorDSL
V = VectorSpace()
x = FreeIndex(V, "x")
A = VariableTensor("A", V)
B = VariableTensor("B", V')
A[x] ⊗ B[x']
# output
IndexSumOperation{0}
    OuterProductOperation{0}
        IndexingOperation{0}
            VariableTensor{1}, A, shape: V_5
            (x,)
        IndexingOperation{0}
            VariableTensor{1}, B, shape: V_5*
            (x*,)
    (x,)
```

## Functions

```@autodocs
Modules = [TensorDSL]
Pages   = ["Operations.jl"]
Order   = [:function]
```

## Trigonometry
```@autodocs
Modules = [TensorDSL]
Pages   = ["Trigonometry.jl"]
Order   = [:function, :type]
```

## Differentation
```@autodocs
Modules = [TensorDSL]
Pages   = ["Differentation.jl"]
```

## Operation nodes

```@autodocs
Modules = [TensorDSL]
Pages   = ["Operations.jl"]
Order   = [:type]
```