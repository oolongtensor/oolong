# TensorDSL
Prototype for a symbolic Tensor DSL in Julia. 

Currently turns tensor operations into an abstract syntax tree.

Example:
```
julia> A = Tensor([1 2 ; 3 4], RnSpace(2), RnSpace(2))
julia> B = VariableTensor(RnSpace(2), RnSpace(2))
julia> x = FreeIndex(RnSpace(2), "x")

julia> (A + B)[1, x]
IndexingOperation{0}
        AddOperation{2}
                Tensor{Int64,2}, [1 2; 3 4], shape: R^2⊗ R^2
                VariableTensor{2}, shape: R^2⊗ R^2
        (1, x)
```

# Supported operations

## Vector Spaces
Vector spaces with or without a known number of dimensions.
```
V4 = VectorSpace(4)
V = VectorSpace()
```
Real vector spaces
```
R2 = RnSpace(2)
```
Dual spaces
```
julia> typeof(V4')
DualVectorSpace
```

## Tensors
Tensors created from multidimensional array

```
A = Tensor([1 2 ; 3 4], R2, R2)
```

Tensors with unknown value
```
B = VariableTensor(R2, R2)
```

Tensors where all entries have the same value
```
julia> C = ConstantTensor(3, V4, V, V4')
ConstantTensor{Int64,3}, 3, shape: V4_2⊗ V_3⊗ V4_2*
```
Zero and and identity tensors
```
Z = ZeroTensor(V4)
I = DeltaTensor(V4, R2)
```

## Tensor operations
Addition
```
julia> A + B
AddOperation{2}
        Tensor{Int64,2}, [1 2; 3 4], shape: R^2⊗ R^2
        VariableTensor{2}, shape: R^2⊗ R^2
```

Outer product
```
julia> A ⊗ C
OuterProductOperation{5}
        Tensor{Int64,2}, [1 2; 3 4], shape: R^2⊗ R^2
        ConstantTensor{Int64,3}, 3, shape: V4_2⊗ V_3⊗ V4_2*
```
Indexing
```
julia> x = FreeIndex(V4, "x")
julia> y = FreeIndex(V, "y")
julia> C[1, y, x']
IndexingOperation{0}
        ConstantTensor{Int64,3}, 3, shape: V4_2⊗ V_3⊗ V4_2*
        (1, y, x*)
```
Component tensor
```
julia>C[1, y]
ComponentTensorOperation{1}
        IndexingOperation{0}
                ConstantTensor{Int64,3}, 3, shape: V4_2⊗ V_3⊗ V4_2*
                (1, y, FreeIndex*)
        (FreeIndex*,)
julia> componenttensor(C[1, y, x'], x', y)
ComponentTensorOperation{2}
        IndexingOperation{0}
                ConstantTensor{Int64,3}, 3, shape: V4_2⊗ V_3⊗ V4_2*
                (1, y, x*)
        (x*, y)
```
Tensor contraction
```
julia> C[1, y, x']*Z[x]
IndexSumOperation{0}
        OuterProductOperation{0}
                IndexingOperation{0}
                        ConstantTensor{Int64,3}, 3, shape: V4_2⊗ V_3⊗ V4_2*
                        (1, y, x*)
                IndexingOperation{0}
                        ZeroTensor{1}, shape: V4_2
                        (x,)
        (x*,)
```
