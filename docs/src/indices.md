## Indices

Indices in Oolong are tied to particular [`Vector Spaces`](@ref). When creating an index,
one must also specify the vector space they are indexing.

Indices on their own are not nodes in AST. Instead, a tuple of indices can be
stored in an [`Indices`](@ref)-node. In most cases the user does not need to create
this, as the index-operations create their own Indices-objects.

```@autodocs
Modules = [TensorDSL]
Pages   = ["Indices.jl"]
```