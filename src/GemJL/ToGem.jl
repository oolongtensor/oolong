function _togem(A::Tensor{T}) where T<: Number
    return LiteralGemTensor(A.value)
end

function togem(node::Node)
    return traversal(node, x-> x, _togem, nothing, nothing)
end