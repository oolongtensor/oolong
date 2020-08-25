function _togem(A::Tensor{T}) where T<:Number
    return gem.Literal(A.value)
end

function _togem(A::ConstantTensor{T}) where T<:Number
    # TODO this does not work if any vector space has unknown dimension
    return gem.Literal(fill(A.value, tuple([dim(V) for V in A.shape]...)))
end

#=function _togem(A::DeltaTensor)
    return IdentityGemTensor([dim(V) for V in A.shape]...)
end=#

function _togem(A::ZeroTensor)
    return gem.Zero(tuple([dim(V) for V in A.shape]...))
end

function _togem(A::VariableTensor)
    return gem.Variable(A.name, tuple([dim(V) for V in A.shape]...))
end

GemIndexTypes = Union{Int, PyObject}

function _togem(in::IndexingOperation, A::PyObject, indices::Tuple{Vararg{GemIndexTypes}})
    return gem.Indexed(A, indices)
end

freeIndices = Dict{FreeIndex, PyObject}()

function _togem(i::FreeIndex)
    global freeIndices
    # Gem gives indices their owns ids, and we want them to be consistent
    if haskey(freeIndices, i)
        return freeIndices[i]
    else
        gemindex = gem.Index(i.name, dim(i.V))
        freeIndices[i] = gemindex
        return gemindex
    end
end

function _togem(i::FixedIndex)
    # Python starts indexing from 0 and Julia from 1
    return i.value - 1
end

function _togem(indices::Indices)
    return tuple([_togem(i) for i in indices.indices]...)
end

```Helper function that indexes tensors. Returns the tensors and the indices.
```
function _toscalar(tensors::Tuple{Vararg{PyObject}})
    indices = []
    for i in 1:length(tensors[1].shape)
        push!(indices, gem.Index(nothing, tensors[1].shape[i]))
    end
    return [gem.Indexed(tensor, indices) for tensor in tensors], indices
end

function _toscalar(tensor::PyObject)
    scalars, indices = _toscalar((tensor,))
    return scalars[1], indices
end

function _togem(add::AddOperation, children::Vararg{PyObject})
    indexed, indices = _toscalar(children)
    sum = indexed[1]
    for expr in indexed[2:end]
        sum += expr
    end
    return gem.ComponentTensor(sum, indices)
end
#=
function _togem(ou::OuterProductOperation, A::ScalarGem, B::ScalarGem)
    return ProductGem(A, B)
end=#

function _togem(comp::ComponentTensorOperation, expr::PyObject, indices::Tuple{Vararg{PyObject}})
    if expr.shape != ()
        expr, new_indices = _toscalar(expr)
        return gem.ComponentTensor(expr, (new_indices..., indices...))
    else
        return gem.ComponentTensor(expr, indices)
    end
end
#=
function _togem(is::IndexSumOperation, expr::ScalarGem, indices::Tuple{Vararg{GemIndex}})
    for i in indices
        expr = IndexSumGem(expr, i)
    end
    return expr
end

_count = 0

```Takes a set of tensors and indices them by the same set of indices. Returns
a list of the indexed tensors and the indices.
```
function _indextensors(tensors::Vararg{GemTensor{rank}}) where rank
    global _count
    indices = []
    for i in 1:rank
        push!(indices, GemIndex(shape(tensors[1])[i], "togem", _count))
        _count += 1
    end
    return [IndexedGem(tensor, indices...) for tensor in tensors], indices
end

function _togem(add::AddOperation, children::Vararg{GemTensor{rank}}) where rank
    indexed, indices = _indextensors(children...)
    return ComponentTensorGem(SumGem(indexed...),  indices...)
end

function _togem(op::OuterProductOperation, A::ScalarGem, B::GemTensor{rank}) where rank
    indexed, indices = _indextensors(B)
    return ComponentTensorGem(ProductGem(A, indexed...), indices...)
end

function _togem(op::OuterProductOperation, A::GemTensor{rank}, B::ScalarGem) where rank
    indexed, indices = _indextensors(A)
    return ComponentTensorGem(ProductGem(indexed..., B), indices...)
end

function _togem(op::OuterProductOperation, A::GemTensor{rank}, B::GemTensor{rank}) where rank
    indexedA, indicesA = _indextensors(A)
    indexedB, indicesB = _indextensors(B)
    return ComponentTensorGem(ProductGem(indexedA..., indexedB...),  indicesA..., indicesB...)
end

function _togem(comp::ComponentTensorOperation, A::ComponentTensorGem{rank},
        indices::Tuple{Vararg{GemIndex}}) where rank
    return ComponentTensorGem(A.children[1], tuple(union(A.children[1].indices, indices)...)...)
end

function _togem(is::IndexSumOperation, A::ComponentTensorGem{rank},
        indices::Tuple{Vararg{GemIndex}}) where rank
    indexed, componentindices = _indextensors(A)
    A = indexed[1]
    for i in indices
        A = IndexSumGem(A, i)
    end
    return ComponentTensorGem(A, componentindices...)
end
=#

function _togem(root::RootNode, node)
    return RootNode(node)
end

function togem(node::Node)
    return traversal(node, x-> x, _togem, nothing, nothing)
end