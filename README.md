# TensorDSL
Prototype for a symbolic Tensor DSL in Julia. 

Currently turns tensor operations into an abstract syntax tree.

Example:
```
julia> A = Tensor([1 2 ; 3 4], RnSpace(2), RnSpace(2))
julia> B = Tensor([5 6 ; 7 8], RnSpace(2), RnSpace(2))

julia> (A + B)[1, 2]
IndexingOperation{0}
        AddOperation{2}
                Tensor{Int64,2}, [1 2; 3 4], shape: R^2⊗ R^2
                Tensor{Int64,2}, [5 6; 7 8], shape: R^2⊗ R^2
        (1, 2)
```
