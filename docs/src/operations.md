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
# Differentiation
```@docs
differentiate(A::AbstractTensor{0}, x::VariableTensor{0})
divergence(A::AbstractTensor{1}, vars::Vararg{VariableTensor{0}})
```
# Trigonometry
```@docs
Base.sin(x)
Base.cos(x)
Base.tan(x)
Base.asin(x)
Base.acos(x)
Base.atan(x)
SineOperatin
CosineOperation
TangentOperation
ArcsineOperation
ArccosineOperation
ArctangentOperation
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