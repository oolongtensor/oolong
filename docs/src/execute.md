## Compilation

Oolong is compiled using [FireDrake's](https://firedrakeproject.org/)
infrastructure.

First Oolong-nodes can be translated to [GEM](https://github.com/firedrakeproject/tsfc/tree/master/gem),
which is a part of [TSFC](https://epubs.siam.org/doi/pdf/10.1137/17M1130642). The function [`togem(node::Node)`](@ref)
translates a given Oolong-node to GEM. The GEM is then futher compiled to [Impero](https://github.com/firedrakeproject/tsfc/blob/master/gem/impero.py)
and from there to [Loopy](https://github.com/firedrakeproject/loopy). Loopy is then compiled to C-code and executed
using [PyOp2](https://github.com/OP2/PyOP2).

In most cases the user does not need to do this directly. They can create a
PyOp2 [`Kernel`](@ref) and execute it with the function [`execute(knl::Kernel, variables::Dict{String, T}) where T`](@ref).
If they want, they can also execute the node directly with the [`execute(knl::Kernel, variables::Dict{String, T}) where T`](@ref)-function.
```@autodocs
Modules = [TensorDSL]
Pages   = ["GenerateCode.jl", "togem.jl"]
```