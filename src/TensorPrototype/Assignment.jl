# Can't use replace(Array, Pair) because of #22
function replaceshape(A::Node, pair::Pair)
    dualspaces = [i == first(pair)' ? last(pair)' : i for i in A.shape]
    return tuple([i == first(pair) ? last(pair) : i for i in dualspaces]...)
end

Assignment = Union{Pair{VariableTensor{rank}, T},
    Pair{T1,T2},
    Pair{FreeIndex{T1}, FixedIndex{T1}}} where {rank, T<:AbstractTensor{rank}, T1 <: AbstractVectorSpace, T2 <: AbstractVectorSpace}

function _assign(visited, A::Tensor{T, rank}, pair::Pair{T1, T2}) where {T, rank, T1<:AbstractVectorSpace, T2<:AbstractVectorSpace}
    return Tensor(A.value, replaceshape(A, pair)...)
end

function _assign(visited, A::ConstantTensor{rank}, pair::Pair{T1, T2}) where {rank, T1<:AbstractVectorSpace, T2<:AbstractVectorSpace}
    return ConstantTensor(A.value, replaceshape(A, pair)...)
end

function _assign(visited, A::ZeroTensor{rank}, pair::Pair{T1, T2}) where {rank, T1<:AbstractVectorSpace, T2<:AbstractVectorSpace}
    return ZeroTensor(replaceshape(A, pair)...)
end

function _assign(visited, A::DeltaTensor{rank}, pair::Pair{T1, T2}) where {rank, T1<:AbstractVectorSpace, T2<:AbstractVectorSpace}
    return DeltaTensor(replaceshape(A, pair)...)
end

function _assign(visited, A::VariableTensor{rank}, pair::Pair{T1, T2}) where {rank, T1<:AbstractVectorSpace, T2<:AbstractVectorSpace}
    return VariableTensor(A.name, replaceshape(A, pair)...)
end

function _assign(visited, indices::Indices, pair::Pair{T1, T2}) where {T1<:AbstractVectorSpace, T2<:AbstractVectorSpace}
    return Indices([i.V == first(pair) ? updatevectorspace(i, last(pair)) : i for i in indices.indices]...)
end

function _assign(visited, indices::Indices, pair::Pair{FreeIndex{T}, FixedIndex{T}}) where {T<:AbstractVectorSpace}
    return Indices([i == first(pair) ? last(pair) : i for i in indices.indices]...)
end

function _assign(visited, A::VariableTensor, pair::Pair{VariableTensor{rank},
        T}) where {rank, T<:AbstractTensor{rank}}
    return A == first(pair) ? last(pair) : A
end

function _assign(visited, A::ConstantTensor, pair::Pair{VariableTensor{0}, S}) where {S<: Scalar}
    return A.value == first(pair) ? ConstantTensor(last(pair), A.shape...) : A
end

function _assign(visited, s::VariableTensor{0}, pair::Pair{VariableTensor{0}, S}) where {S<: Scalar}
    return s == first(pair) ? last(pair) : s
end

function _assign(visited, A::Tensor, pair::Pair{VariableTensor{0}, S}) where {S<: Scalar}
    function assigninarray(x, pair)
        if last(pair) isa ConstantTensor && x == first(pair) 
            return last(pair).value
        elseif last(pair) isa ZeroTensor && x == first(pair)
            return 0
        else
            return assign(x, pair)
        end
    end  
    return Tensor(assigninarray.(A.value, pair), A.shape...)
end

function _assign(visited, x::Number, pair::Assignment, children::Vararg{Node})
    return x
end

function _assign(visited, x::Node, pair::Assignment, children::Vararg{Node})
    return updatechildren(x, children...)
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

function assign(node::Node, pair::Pair{T, N}) where {T<:AbstractTensor{0}, N<:Number}
    return traversal(node, x-> x, _assign, nothing, first(pair)=>Tensor(last(pair)))
end

function assign(node::Node, pair::Pair{FreeIndex{T}, FixedIndex{T}}) where {T<:AbstractVectorSpace}
    if first(pair).V != last(pair).V
        throw(Dimensionmismatch(string(first(pair).V, "!=", last(pair).V)))
    end
    return traversal(node, x-> x, _assign, nothing, pair)
end

function assign(node::Node, pair::Pair{FreeIndex{T}, Int}) where {T<:AbstractVectorSpace}
    return assign(node, first(pair)=>FixedIndex(first(pair).V, last(pair)))
end

function assign(node::Node, pair::Assignment)
    return traversal(node, x-> x, _assign, nothing, pair)
end

function assign(i::Number, pair::Assignment)
    return i
end