include("Node.jl")

abstract type AbstractIndex end

struct ConcreteIndex <: AbstractIndex
    val
end
