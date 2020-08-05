include("Tensors.jl")
include("Indices.jl")
include("../TreeVisitor/Traversal.jl")

# Can't use replace(Array, Pair) because of #22
function replaceshape(A::Node, pair::Pair)
    return tuple([i == first(pair) ? last(pair) : i for i in A.shape]...)
end

Assignment = Union{Pair{VariableTensor{rank}, Tensor{T, rank}},
    Pair{ScalarVariable, Scalar},
    Pair{T1,T2}} where {T, rank, T1 <: AbstractVectorSpace, T2 <: AbstractVectorSpace}

function _assign(A::Union{Tensor{T, rank}, ConstantTensor{T, rank}}, pair::Pair{T1, T2}) where {T, rank, T1<:AbstractVectorSpace, T2<:AbstractVectorSpace}
    return Tensor(A.value, replaceshape(A, pair)...)
end

function _assign(A::Union{VariableTensor{rank}, ZeroTensor{rank}, DeltaTensor{rank}}, pair::Pair{T1, T2}) where {rank, T1<:AbstractVectorSpace, T2<:AbstractVectorSpace}
    return VariableTensor(replaceshape(A, pair)...)
end

function _assign(A::VariableTensor, pair::Pair{VariableTensor{rank},
        Tensor{T, rank}}) where {T, rank}
    if A == first(pair)
        return last(pair)
    else
        return A
    end
end

function _assign(node::Node, pair::Assignment)
    return node
end

function assign(node::Node, pair::Pair{VariableTensor{rank}, Tensor{T, rank}}) where {rank, T}
    # Check whether we need to assign vector spaces
    vsassignments = Dict{AbstractVectorSpace,AbstractVectorSpace}()
    A, B = pair
    for i in 1:length(A.shape)
        if A.shape[i] != B.shape[i]
            if dim(A.shape[i]) !== nothing && get(vsassignments, A.shape[i], nothing) != B.shape[i]
                throw(DomainError((A.shape[i], B.shape[i]), string("Invalid reshaping of ", A.shape, " to ", B.shape)))
            else
                vsassignments[A.shape[i]] = B.shape[i]
            end
        end
    end
    if isempty(vsassignments)
        pretraversalargs = nothing
    else
        pretraversalargs = tuple([kv for kv in vsassignments]...)
    end
    return traversal(node, assign, _assign, nothing, pair)
end

function assign(node::Node, pairs::Vararg{Assignment})
    for pair in pairs
        node = traversal(node, x->x, _assign, nothing, pair)
    end
    return node
end