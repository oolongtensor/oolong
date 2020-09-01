"""Abstract index supertype"""
abstract type Index end

"""An index whose value is not known. Identified by its name. There should be
no need to set the id by the user.
"""
struct FreeIndex{T<:AbstractVectorSpace} <: Index
    V::T
    name::String
    id::Int
end

FreeIndex(V::T, name::String) where {T<:AbstractVectorSpace} = FreeIndex(V, name, 0)

"""Creates an index with the same name in the dual space."""
Base.adjoint(i::FreeIndex) = FreeIndex(dual(i.V), i.name, i.id)

"""An index whose value is known."""
struct FixedIndex{T<:AbstractVectorSpace} <: Index
    V::T
    value::Int
    function FixedIndex(V::T, value::Int) where {T<:AbstractVectorSpace}
        if dim(V) == nothing || value < 1 || value > dim(V)
            throw(DomainError(value, string("Value not in range for the vectorspace ", V)))
        end
        new{T}(V, value)
    end
end

"""Creates an index with the same value in the dual space."""
Base.adjoint(i::FixedIndex) = FixedIndex(dual(i.V), i.value)

"""A node for holding indices."""
struct Indices <: Node
    indices
    children
end

Indices(indices::Vararg{Index}) = Indices(indices, ())

"""Creates an index from an integer and VectorSpace."""
function toindex(V::AbstractVectorSpace, i::Int)
    return FixedIndex(V, i)
end

"""Creates an index from a string and VectorSpace."""
function toindex(V::AbstractVectorSpace, s::String)
    return FreeIndex(V, s)
end

"""Added for convenience."""
function toindex(V::AbstractVectorSpace, i::Index)
    return i
end

Base.show(io::IO, indices::Indices) = print(io, indices.indices)

function Base.show(io::IO, i::FreeIndex)
    if i.name != ""
        print(io, i.name)
    else
        print(io, "FreeIndex")
    end
    if i.V isa DualVectorSpace
        print(io, "*")
    end
end

Base.show(io::IO, i::FixedIndex) = print(io, i.value)
