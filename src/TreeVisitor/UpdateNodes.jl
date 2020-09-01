function updatechildren(node::Node)
    return node
end

updatechildren(root::RootNode, node::Node) = RootNode(node)

### Operations ###

function updatechildren(indexsum::IndexSumOperation, A::AbstractTensor, indices::Indices)
    return contractioncheck(A)
end

function updatechildren(add::AddOperation, children::Vararg{Node})
    return +(children...)
end

function updatechildren(op::OuterProductOperation, A::Node, B::Node)
    return A⊗B
end

function updatechildren(indexing::IndexingOperation, A::AbstractTensor, indices::Indices)
    return A[indices.indices...]
end

function updatechildren(comp::ComponentTensorOperation, A::AbstractTensor, indices::Indices)
    return componenttensor(A, indices.indices...)
end

### Trigonometry ###

function updatechildren(si::SineOperation, A::AbstractTensor)
    return Base.sin(A)
end

function updatechildren(co::CosineOperation, A::AbstractTensor)
    return Base.cos(A)
end

function updatechildren(ta::TangentOperation, A::AbstractTensor)
    return Base.tan(A)
end

## Indices
function updatevectorspace(i::FreeIndex, V::AbstractVectorSpace)
    return FreeIndex(V, i.name, i.id)
end

function updatevectorspace(i::FixedIndex, V::AbstractVectorSpace)
    return FixedIndex(V, i.value)
end