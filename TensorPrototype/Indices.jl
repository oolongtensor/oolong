include("Node.jl")

abstract type AbstractIndices <: Node end

struct FreeIndex
    name
end

Index = Union{Int, FreeIndex}

struct ConcreteIndices <: AbstractIndices
    indices
    children
end

ConcreteIndices(indices::Vararg{Index}) = ConcreteIndices(indices, ())
