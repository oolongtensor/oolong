include("Node.jl")
include("VectorSpace.jl")

abstract type Index end

struct FreeIndex{T<:AbstractVectorSpace} <: Index
    V::T
    name::String
    id::Int
end

FreeIndex(V::T, name::String) where {T<:AbstractVectorSpace} = FreeIndex(V, name, 0)

Base.adjoint(i::FreeIndex) = FreeIndex(dual(i.V), i.name, i.id)

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

Base.adjoint(i::FixedIndex) = FixedIndex(dual(i.V), i.value)

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

# Added for convenience, does nothing
function toindex(V::AbstractVectorSpace, i::Index)
    return i
end
