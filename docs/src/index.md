# Oolong
Prototype for a symbolic Tensor DSL in Julia. 

Turns tensor operations into an abstract syntax tree.

Example:
```jldoctest
using TensorDSL
A = Tensor([1 2 ; 3 4], RnSpace(2), RnSpace(2))
B = VariableTensor("B", RnSpace(2), RnSpace(2))
x = FreeIndex(RnSpace(2), "x")

(A + B)[1, x]

# output

IndexingOperation{0}
    AddOperation{2}
        Tensor{Int64,2}, [1 2; 3 4], shape: R^2⊗ R^2
        VariableTensor{2}, B, shape: R^2⊗ R^2
    (1, x)
```
