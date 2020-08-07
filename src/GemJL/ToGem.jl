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

function _togem(add::AddOperation, children::Vararg{ScalarGem})
    return SumGem(children...)
end

_count = 0

function _togem(add::AddOperation, children::Vararg{GemTensor{rank}}) where rank
    global _count
    indices = []
    for i in 1:rank
        push!(indices, GemIndex(shape(children[1])[i], "togem", _count))
        _count += 1
    end
    return SumGem([IndexedGem(child, indices...) for child in children]...)
end

function togem(node::Node)
    return traversal(node, x-> x, _togem, nothing, nothing)
end