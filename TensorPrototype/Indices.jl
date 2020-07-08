include("Node.jl")
include("VectorSpace.jl")

abstract type AbstractIndices <: Node end

abstract type Index end

struct FreeIndex <: Index
    V::AbstractVectorSpace
end

Base.adjoint(i::FreeIndex) = FreeIndex(dual(i.V))

struct FixedIndex <: Index
    value::Int
    V::AbstractVectorSpace
    function FixedIndex(value::Int, V::AbstractVectorSpace)
        if value < 1 || value > dim(V)
            error("Index not in range")
        end
        new(value, V)
    end
end

Base.adjoint(i::FixedIndex) = FixedIndex(i.value, dual(i.V))

struct Indices <: AbstractIndices
    indices
    children
end

Indices(indices::Vararg{Index}) = Indices(indices, ())
