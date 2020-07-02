include("Node.jl")
include("Tensors.jl")

abstract type Operation <: Node end

# Should I instead have a single operation type with the operation type defined?
struct Add <: Operation
    children::Tuple{Vararg{Node}}
end

function +(x::Scalar, y::Scalar)
    return Add((ConcreteTensor(x), ConcreteTensor(y)))
end
