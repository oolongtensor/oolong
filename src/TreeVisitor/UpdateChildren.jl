include("../TensorPrototype/Differentation.jl")

function updateChildren(node::Node)
    return node
end

function updateChildren(indexsum::IndexSumOperation, A::AbstractTensor, indices::Indices)
    return contractioncheck(A)
end

function updateChildren(add::AddOperation, children::Vararg{Node})
    return +(children...)
end

function updateChildren(op::OuterProductOperation, A::Node, B::Node)
    return A⊗B
end

function updateChildren(indexing::IndexingOperation, A::AbstractTensor, indices::Indices)
    return A[indices.indices...]
end

function updateChildren(comp::ComponentTensorOperation, A::AbstractTensor, indices::Indices)
    return componenttensor(A, indices...)
end
