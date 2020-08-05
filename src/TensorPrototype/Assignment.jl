include("Tensors.jl")
include("Indices.jl")
include("../TreeVisitor/UpdateChildren.jl")

Assignment = Union{Pair{VariableTensor{rank}, Tensor{T, rank}},
    Pair{ScalarVariable, Scalar},
    Pair{VectorSpace, VectorSpace}} where {T, rank}

function assign(A::VariableTensor, pair::Pair{VariableTensor{rank},
        Tensor{T, rank}}) where {T, rank}
    if A == first(pair)
        return last(pair)
    else
        return A
    end
end

function assign(node::Node, pair::Assignment) where {T, rank}
    new_children = [assign(child, pair) for child in node.children]
    return updateChildren(node, tuple(new_children...)...)
end