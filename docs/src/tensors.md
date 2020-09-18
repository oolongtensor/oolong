# Tensors

The tensor product of two vector spaces ``E`` and ``F`` is a new vector space,
denoted by ``E⊗F``. It is spanned by the elements ``e⊗f`` where
``e \in E= e_i \phi_i`` and  ``f \in F= f_j \psi_j`` then
``g \in E ⊗ F= g_{ij} \phi_i ⊗ \psi_j``.

Tensors are these objects in the vector spaces created by tensors products

Mathematically, tensor ``T`` is ``T \in E_1⊗…⊗E_p⊗F_1^∗⊗…⊗F_q^∗`` (or a permutation of these),
where ``E_i`` and ``F_j^∗`` are vector and dual spaces, respectively
Programmatically this means that tensors objects with shape and data. Shape
describes the tensor space ``E_1⊗…⊗E_p⊗F_1^∗⊗…⊗F_q^∗`` and data describes the
vector within this tensor space.


There are two main types of tensors in Oolong: tensors created from
multidimensional arrays:

```jldoctest
using TensorDSL
R2 = RnSpace(2)
A = Tensor([1 2 ; 3 4], R2, R2)

# output

Tensor{Int64,2}, [1 2; 3 4], shape: R^2⊗ R^2
```

and tensors with unknown value
```jldoctest
using TensorDSL
R2 = RnSpace(2)
B = VariableTensor("B", R2, R2)

# output

VariableTensor{2}, B, shape: R^2⊗ R^2
```

Zero and and identity tensors have their own types.
```jldoctest; setup=(using TensorDSL)
julia> Z = ZeroTensor(RnSpace(3))
ZeroTensor{1}, shape: R^3
julia> I = DeltaTensor(RnSpace(3), RnSpace(3))
DeltaTensor{2}, shape: R^3⊗ R^3
```

There is also a type for tensors where all entries have the same value.
```jldoctest; setup=(using TensorDSL)
julia> C = ConstantTensor(3)
ConstantTensor{Int64,0}, 3, shape: ()
```

```@autodocs
Modules = [TensorDSL]
Pages   = ["Tensors.jl"]
```