include("Node.jl")

abstract type AbstractIndex <: Node end

struct FreeIndex
    name
end

Index = Union{Int, FreeIndex}

struct ConcreteIndex <: AbstractIndex
    indices
    children
end

ConcreteIndex(indices...) = ConcreteIndex(indices, ())
