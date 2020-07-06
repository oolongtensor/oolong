include("Node.jl")
include("Tensors.jl")
include("Indices.jl")

import Base

abstract type Operation <: AbstractTensor end

struct Add <: Operation
    shape
    children::Tuple{Vararg{AbstractTensor}}
end

function +(nodes::Vararg{Node})
    if length(nodes) > 1
        for node in nodes[2:length(nodes)]
            if node.shape != nodes[1].shape
                # TODO: Better error message
                error("Shapes don't match")
            end
        end
    end
    return Add(nodes[1].shape, nodes)
end

struct IndexingOperation <: Operation
    shape
    children::Tuple{AbstractIndices, AbstractTensor}
    function IndexingOperation(x::Node, y::AbstractIndices)
        shapearray = []
        if length(y.indices) > length(x.shape)
            error("Too many indices")
        end
        for i in 1:length(x.shape)
            if i > length(y.indices) || typeof(y.indices[i]) == Colon
                push!(shapearray, x.shape[i])
            elseif typeof(y.indices[i]) <: Int
                if y.indices[i] > x.shape[i]
                    error("Index out of range")
                end
                # TODO Check for free indices
            end
        end
        shapetuple = tuple(shapearray...)
        if length(shapetuple) < 1
            shapetuple = (1,)
        end
        new(shapetuple, (y, x))
    end
end

function Base.getindex(x::AbstractTensor, y::AbstractIndices)
    return IndexingOperation(x, y)
end

function Base.getindex(x::AbstractTensor, ys::Vararg{Index})
    return IndexingOperation(x, ConcreteIndices(ys...))
end
