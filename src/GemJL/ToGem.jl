function _togem(A::Tensor{T}) where T<: Number
    return LiteralGemTensor(A.value)
end

function _togem(root::RootNode, child::Vararg{Node})
    return updatechildren(root, child...)
end

function _togem(add::AddOperation, children::Vararg{ScalarGem})
    return AddGem(children)
end

function togem(node::Node)
    return traversal(node, x-> x, _togem, nothing, nothing)
end