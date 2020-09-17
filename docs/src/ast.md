# AST

Oolong generates an abstract syntax tree of tensor operations. Each operation
on tensors is translated to an operation node.

```jldoctest
using TensorDSL
A = VariableTensor("A")
B = VariableTensor("B")
A + 3*B

# output

AddOperation{0}
    VariableTensor{0}, A, shape: ()
    OuterProductOperation{0}
        ConstantTensor{Int64,0}, 3, shape: ()
        VariableTensor{0}, B, shape: ()
```

This tree can then be traversed using traversal-function.
Examples of this include [`togem(node::Node)`](@ref) and [`differentiate(A::AbstractTensor{0}, x::VariableTensor{0})`](@ref).

The function [`assign(node::Node, pair::Pair{VariableTensor{rank}, T}) where {rank, T<:AbstractTensor{rank}}`](@ref)
can be used to assign values for the variables in the tree.

```@autodocs
Modules = [TensorDSL]
Pages   = ["Node.jl", "Traversal.jl", "Assignment.jl"]
```