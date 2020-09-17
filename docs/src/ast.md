## AST

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

This tree can then be traversed using [`traversal(node, pretraversalfn::Function, visitfn::Function,
        pretraversalfnargs::Union{Any, Nothing}, visitfnargs::Union{Any, Nothing},
        visited::Union{Dict, Nothing}, posttraversal=false)`](@ref)-function.
Examples of this include [`togem(node::Node)`](@ref) and [`differentiate(A::AbstractTensor{0}, x::VariableTensor{0})`].

```@autodocs
Modules = [TensorDSL]
Pages   = ["Traversal.jl"]
```