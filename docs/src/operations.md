## Functions

```@docs
Base.:+(nodes::Vararg{Node})
⊗(A::AbstractTensor, B::AbstractTensor)
Base.:*(x::Scalar, A::AbstractTensor)
Base.:-(A::AbstractTensor)
Base.:-(A::AbstractTensor{rank}, B::AbstractTensor{rank}) where rank
Base.:^(x::AbstractTensor{0}, y::AbstractTensor{0})
Base.:^(x::AbstractTensor{0}, y::Number)
Base.sqrt(x::AbstractTensor{0})
Base.:/(A::AbstractTensor, y::AbstractTensor{0})
Base.:/(y::Number, A::AbstractTensor{0})
componenttensor(A::AbstractTensor, indices::Vararg{Index})
Base.getindex(A::AbstractTensor, ys::Vararg{Index})
Base.getindex(A::AbstractTensor, ys::Vararg{Union{String, Int, Index}})
Base.transpose(A::AbstractTensor)
trace(A::AbstractTensor{2})
```

## Operation Nodes

```@docs
Operation
AddOperation
OuterProductOperation
DivisionOperation
PowerOperation
ComponentTensorOperation
IndexingOperation
IndexSumOperation
```