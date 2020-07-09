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
    function FixedIndex(V::T, value::Int) where {T<:AbstractVectorSpace}
        if value < 1 || value > dim(V)
            error("Index not in range")
        end
        new{T}(V, value)
    end
end

Base.adjoint(i::FixedIndex) = FixedIndex(i.value, dual(i.V))

struct Indices <: Node
    indices
    children
end

Indices(indices::Vararg{Index}) = Indices(indices, ())

function toindex(V::AbstractVectorSpace, i::Int)
    return FixedIndex(V, i)
end

function toindex(V::AbstractVectorSpace, s::String)
    return FreeIndex(V, s)
end

function toindex(i::Index)
    return i
end
