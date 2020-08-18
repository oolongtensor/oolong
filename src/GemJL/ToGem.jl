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

function _togem(A::VariableTensor)
    return VariableGemTensor([dim(V) for V in A.shape]...)
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

function _togem(comp::ComponentTensorOperation, expr::ScalarGem, indices::Tuple{Vararg{GemIndex}})
    return ComponentTensorGem(expr, indices...)
end

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
    indexed, indices = _indextensors(A)
    return ComponentTensorGem(IndexSumOperation(indexed,
        tuple(union(A.children[1].indices, indices)...)...))
end


function togem(node::Node)
    return traversal(node, x-> x, _togem, nothing, nothing)
end