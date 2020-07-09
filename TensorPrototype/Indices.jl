include("Node.jl")
include("VectorSpace.jl")

abstract type Index end

struct FreeIndex{T<:AbstractVectorSpace} <: Index
    V::T
    name::String
end

Base.adjoint(i::FreeIndex) = FreeIndex(i.name, dual(i.V))

struct FixedIndex{T<:AbstractVectorSpace} <: Index
    V::T
    value::Int
    function FixedIndex{T}(value::Int, V::T) where {T<:AbstractVectorSpace}
        if value < 1 || value > dim(V)
            error("Index not in range")
        end
        new(value, V)
    end
end

Base.adjoint(i::FixedIndex) = FixedIndex(i.value, dual(i.V))

struct Indices <: Node
    indices
    children
end

Indices(indices::Vararg{Index}) = Indices(indices, ())

function toindex(i::Int, V::AbstractVectorSpace)
    return FixedIndex(i, V)
end

function toindex(s::String, V::AbstractVectorSpace)
    return FreeIndex(s, V)
end
