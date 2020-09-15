```@docs
Base.:+(nodes::Vararg{Node})
```

```@docs
⊗(A::AbstractTensor, B::AbstractTensor)
```

```@docs
Base.:*(x::Scalar, A::AbstractTensor)
```

```@docs
Base.:-(A::AbstractTensor)
```

```@docs
Base.:-(A::AbstractTensor{rank}, B::AbstractTensor{rank}) where rank
```

```@docs
Base.:^(x::AbstractTensor{0}, y::AbstractTensor{0})
```

```@docs
Base.:^(x::AbstractTensor{0}, y::Number)
```

```@docs
Base.sqrt(x::AbstractTensor{0})
```

```@docs
Base.:/(A::AbstractTensor, y::AbstractTensor{0})
```

```@docs
Base.:/(y::Number, A::AbstractTensor{0})
```

```@docs
componenttensor(A::AbstractTensor, indices::Vararg{Index})
```

```@docs
Base.getindex(A::AbstractTensor, ys::Vararg{Index})
```

```@docs
Base.getindex(A::AbstractTensor, ys::Vararg{Union{String, Int, Index}})
```

```@docs
Base.transpose(A::AbstractTensor)
```

```@docs
trace(A::AbstractTensor{2})
```