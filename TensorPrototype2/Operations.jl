include("Node.jl")
include("Tensors.jl")

abstract type Operation <: Node end

struct Add <: Operation
    shape
    children::Tuple{Vararg{Node}}
end

function +(x::Node, y::Node)
    if x.shape != y.shape
        # TODO: Better error message
        error("Shapes don't match")
    end
    return Add(x.shape, (x, y))
end
