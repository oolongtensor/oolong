function _togem(A::Tensor{T}) where T<: Number
    return LiteralGemTensor(A.value)
end

function _togem(root::RootNode, child::Vararg{Node})
    return updatechildren(root, child...)
end

function togem(node::Node)
    return traversal(node, x-> x, _togem, nothing, nothing)
end