include("Tensors.jl")
include("Indices.jl")
include("../TreeVisitor/Traversal.jl")

# Can't use replace(Array, Pair) because of #22
function replaceshape(A::Node, pair::Pair)
    dualspaces = [i == first(pair)' ? last(pair)' : i for i in A.shape]
    return tuple([i == first(pair) ? last(pair) : i for i in dualspaces]...)
end

Assignment = Union{Pair{VariableTensor{rank}, T},
    Pair{ScalarVariable, S},
    Pair{T1,T2}} where {rank, T<:AbstractTensor{rank}, S<:Scalar, T1 <: AbstractVectorSpace, T2 <: AbstractVectorSpace}

function _assign(A::Tensor{T, rank}, pair::Pair{T1, T2}) where {T, rank, T1<:AbstractVectorSpace, T2<:AbstractVectorSpace}
    return Tensor(A.value, replaceshape(A, pair)...)
end

function _assign(A::ConstantTensor{rank}, pair::Pair{T1, T2}) where {rank, T1<:AbstractVectorSpace, T2<:AbstractVectorSpace}
    return ConstantTensor(A.value, replaceshape(A, pair)...)
end

function _assign(A::ZeroTensor{rank}, pair::Pair{T1, T2}) where {rank, T1<:AbstractVectorSpace, T2<:AbstractVectorSpace}
    return ZeroTensor(replaceshape(A, pair)...)
end

function _assign(A::DeltaTensor{rank}, pair::Pair{T1, T2}) where {rank, T1<:AbstractVectorSpace, T2<:AbstractVectorSpace}
    return DeltaTensor(replaceshape(A, pair)...)
end

function _assign(A::VariableTensor{rank}, pair::Pair{T1, T2}) where {rank, T1<:AbstractVectorSpace, T2<:AbstractVectorSpace}
    return VariableTensor(A.name, replaceshape(A, pair)...)
end

function _assign(A::VariableTensor, pair::Pair{VariableTensor{rank},
        T}) where {rank, T<:AbstractTensor{rank}}
    return A == first(pair) ? last(pair) : A
end

function _assign(A::ConstantTensor, pair::Pair{ScalarVariable, S}) where {S<: Scalar}
    return A.value == first(pair) ? ConstantTensor(last(pair), A.shape...) : A
end

function _assign(s::ScalarVariable, pair::Pair{ScalarVariable, S}) where {S<: Scalar}
    return s == first(pair) ? last(pair) : s
end

function _assign(A::Tensor, pair::Pair{ScalarVariable, S}) where {S<: Scalar}
    return Tensor(assign.(A.value, pair), A.shape...)
end

function _assign(x::Union{Node, Number}, pair::Assignment)
    return x
end

function assign(node::Node, pair::Pair{VariableTensor{rank}, T}) where {rank, T<:AbstractTensor{rank}}
    # Check that vector spaces match
    A, B = pair
    for i in 1:length(A.shape)
        if A.shape[i] != B.shape[i]
            throw(DomainError((A.shape[i], B.shape[i]), string("Invalid reshaping of ", A.shape, " to ", B.shape)))
        end
    end
    return traversal(node, x-> x, _assign, nothing, pair)
end

function assign(node::Node, pair::Pair{T1, T2}) where {T1<:AbstractVectorSpace, T2<:AbstractVectorSpace}
    # Check that vector spaces match
    if dim(first(pair)) != dim(last(pair)) && dim(first(pair)) !== nothing
        throw(DomainError((T1,T2), "Cannot assign different dimension vector spaces"))
    end
    return traversal(node, x-> x, _assign, nothing, pair)
end

function assign(node::Node, pair::Assignment)
    return traversal(node, x-> x, _assign, nothing, pair)
end

function assign(i::ScalarVariable, pair::Assignment)
    return _assign(i, pair)
end

function assign(i::Number, pair::Assignment)
    return i
end