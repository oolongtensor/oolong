function _togem(A::Tensor{T}) where T<:Number
    return LiteralGemTensor(A.value)
end

function _togem(A::ConstantTensor{T}) where T<:Number
    return LiteralGemTensor(fill(A.value, tuple([dim(V) for V in A.shape]...)))
end

function _togem(A::DeltaTensor)
    return IdentityGemTensor([dim(V) for V in A.shape]...)
end

function _togem(A::ZeroTensor)
    return ZeroGemTensor([dim(V) for V in A.shape]...)
end

function _togem(root::RootNode, child::Vararg{Node})
    return updatechildren(root, child...)
end

function _togem(in::IndexingOperation, A::GemTensor{rank}, indices::Tuple{Vararg{GemIndexTypes}}) where rank
    return IndexedGem(A, indices...)
end

function _togem(i::FreeIndex)
    return GemIndex(dim(i.V), i.name, i.id)
end

function _togem(i::FixedIndex)
    return i.value
end

function _togem(indices::Indices)
    return tuple([_togem(i) for i in indices.indices]...)
end

function _togem(add::AddOperation, children::Vararg{ScalarGem})
    return SumGem(children...)
end

function _togem(ou::OuterProductOperation, A::ScalarGem, B::ScalarGem)
    return ProductGem(A, B)
end

function _togem(comp::ComponentTensorOperation, expr::ScalarGem, indices::Tuple{Vararg{FreeIndex}})
    return ComponentTensorGem(expr, indices)
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

function togem(node::Node)
    return traversal(node, x-> x, _togem, nothing, nothing)
end