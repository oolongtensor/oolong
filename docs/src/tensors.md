# Tensors

There are two main types of tensors in Oolong: tensors created from
multidimensional arrays:

```
A = Tensor([1 2 ; 3 4], R2, R2)
```

and tensors with unknown value
```
B = VariableTensor(R2, R2)
```

Zero and and identity tensors have their own types.
```
Z = ZeroTensor(V4)
I = DeltaTensor(V4, R2)
```

There is also a type for tensors where all entries have the same value.
```
julia> C = ConstantTensor(3, V4, V, V4')
ConstantTensor{Int64,3}, 3, shape: V4_2⊗ V_3⊗ V4_2*
```

```@autodocs
Modules = [TensorDSL]
Pages   = ["Tensors.jl"]
```