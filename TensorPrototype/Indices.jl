include("Node.jl")
include("VectorSpace.jl")

abstract type Index end

struct FreeIndex{T<:AbstractVectorSpace} <: Index
    V::T
end

Base.adjoint(i::FreeIndex) = FreeIndex(dual(i.V))

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
