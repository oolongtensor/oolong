include("Tensors.jl")
include("Indices.jl")
include("../TreeVisitor/Traversal.jl")

function replaceshape(A::Node, pair::Pair)
    return tuple([i == first(pair) ? last(pair) : i for i in A.shape]...)
end
#=
function updatevectorspace(A::Tensor{T, rank}, pair::Pair{T1, T2}) where {T, rank, T1<:AbstractVectorSpace, T2<:AbstractVectorSpace}
    new_shape = replace([A.shape...], pair)
    return Tensor(A.value, new_shape...)
end=#

function updatevectorspace(A::VariableTensor{rank}, pair::Pair{T1, T2}) where {rank, T1<:AbstractVectorSpace, T2<:AbstractVectorSpace}
    return VariableTensor(replaceshape(A, pair)...)
end

Assignment = Union{Pair{VariableTensor{rank}, Tensor{T, rank}},
    Pair{ScalarVariable, Scalar},
    Pair{VectorSpace, VectorSpace}} where {T, rank}

function _assign(A::VariableTensor, pair::Pair{VariableTensor{rank},
        Tensor{T, rank}}) where {T, rank}
    if A == first(pair)
        return last(pair)
    else
        return A
    end
end

function _assign(A::TerminalTensor, pair::Pair{VectorSpace, VectorSpace})
    return node
end

function _assign(node::Node, pair::Assignment)
    return node
end

function assign(node::Node, pair::Pair{VariableTensor{rank}, Tensor{T, rank}}) where {rank, T}
    return traversal(node, (x -> x), _assign, nothing, pair)
end

function assign(node::Node, pair::Assignment)
    return traversal(node, x->x, _assign, nothing, pair)
end