include("Node.jl")

abstract type AbstractIndices <: Node end

struct FreeIndex
    name
    range
end

Index = Union{Int, FreeIndex, Colon}

struct ConcreteIndices <: AbstractIndices
    indices
    children
end

ConcreteIndices(indices::Vararg{Index}) = ConcreteIndices(indices, ())
