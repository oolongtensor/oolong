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
