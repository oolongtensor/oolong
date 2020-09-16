"""
    Index

Abstract index supertype.
"""
abstract type Index end

"""
    FreeIndex(V::T, name::String, id=0) where {T<:AbstractVectorSpace}

An index whose value is not known. Identified by its name. There should be
no need to set the id by the user.
"""
struct FreeIndex{T<:AbstractVectorSpace} <: Index
    V::T
    name::String
    id::Int
end

FreeIndex(V::T, name::String) where {T<:AbstractVectorSpace} = FreeIndex(V, name, 0)

"""
    FixedIndex(V::T, value::Int) where {T<:AbstractVectorSpace}

An index whose value is known.
"""
struct FixedIndex{T<:AbstractVectorSpace} <: Index
    V::T
    value::Int
    function FixedIndex(V::T, value::Int) where {T<:AbstractVectorSpace}
        if dim(V) === nothing || value < 1 || value > dim(V)
            throw(DomainError(value, string("Value not in range for the vectorspace ", V)))
        end
        new{T}(V, value)
    end
end

"""
    Base.adjoint(i::Index)

Creates an index with the same name/value in the dual space.
"""
Base.adjoint(i::FreeIndex) = FreeIndex(dual(i.V), i.name, i.id)
Base.adjoint(i::FixedIndex) = FixedIndex(dual(i.V), i.value)

"""
    Indices(indices::Vararg{Index})

A node that represents a tuple of indices.
"""
struct Indices <: Node
    indices
    children
end

Indices(indices::Vararg{Index}) = Indices(indices, ())

"""
    toindex(V::AbstractVectorSpace, i::Union{String, Int, Index})

Given a vectorspace V, creates an index of the string or integer i. If i is an
index, returns i.
"""
function toindex(V::AbstractVectorSpace, i::Int)
    return FixedIndex(V, i)
end

function toindex(V::AbstractVectorSpace, s::String)
    return FreeIndex(V, s)
end

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
