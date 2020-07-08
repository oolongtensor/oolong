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
end

Base.adjoint(i::FixedIndex) = FixedIndex(i.value, dual(i.V))

struct Indices <: AbstractIndices
    indices
    children
end

Indices(indices::Vararg{Index}) = Indices(indices, ())
