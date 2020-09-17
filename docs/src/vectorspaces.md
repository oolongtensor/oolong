# Vector Spaces

The shape of [Tensors](@ref) is a represented as a tuple of vector spaces. 

Vector spaces with or without a known number of dimensions. (Note: vector spaces
with unknow dimension cannot be compiled to GEM.)
```
V4 = VectorSpace(4)
V = VectorSpace()
```
Dual spaces
```
julia> typeof(V4')
DualVectorSpace
```
Real vector spaces
```
R2 = RnSpace(2)
```

```@autodocs
Modules = [TensorDSL]
Pages   = ["VectorSpace.jl"]
```