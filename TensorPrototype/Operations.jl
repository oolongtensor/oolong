include("Tensors.jl")
include("Indices.jl")

import Base

abstract type Operation <: AbstractTensor end

struct AddOperation <: Operation
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{Vararg{AbstractTensor}}
    freeindices::Set{FreeIndex}
end

function Base.:+(nodes::Vararg{Node})
    if length(nodes) > 1
        for node in nodes[2:length(nodes)]
            if node.shape != nodes[1].shape
                # TODO: Better error message
                error("Shapes don't match")
            end
        end
    end
    return AddOperation(nodes[1].shape, nodes, union((node.freeindices for node=nodes)...))
end

struct IndexingOperation <: Operation
    shape::Tuple{}
    children::Tuple{AbstractTensor, Indices}
    freeindices::Set{FreeIndex}
end

IndexingOperation(x::AbstractTensor, indices::Indices) = IndexingOperation((),(x, indices), union(Set([i for i=indices.indices if i isa FreeIndex]), x.freeindices))

function Base.getindex(x::AbstractTensor, y::Indices)
    if length(x.shape) != length(y.indices)
        error("Invalid number of indices")
    end
    for i in 1:length(y.indices)
        if x.shape[i] != y.indices[i].V
            error("Invalid vector space")
        end
    end
    return IndexingOperation(x, y)
end

function Base.getindex(x::AbstractTensor, ys::Vararg{Index})
    return Base.getindex(x, Indices(ys...))
end


function Base.getindex(x::AbstractTensor, ys::Vararg{Union{String, Int, Index}})
    indexarray = []
    for i in 1:length(ys)
        push!(indexarray, toindex(x.shape[i], ys[i]))
    end
    return Base.getindex(x, Indices(tuple(indexarray...)...))
end

struct OuterProductOperation <: Operation
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{AbstractTensor, AbstractTensor}
    freeindices::Set{FreeIndex}
end

function ⊗(x::AbstractTensor, y::AbstractTensor)
    return OuterProductOperation(tuple(x.shape..., y.shape...), (x, y))
end

function Base.:*(x::Scalar, A::AbstractTensor)
    return Tensor([x]) ⊗ A
end

function Base.:-(A::AbstractTensor)
    return -1*A
end

function Base.:-(A::AbstractTensor, B::AbstractTensor)
    return A + (-1*B)
end

struct ComponentTensorOperation <: Operation
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{AbstractTensor, Int}
    freeindices::Set{FreeIndex}
end

# function componentTensor(AbstractTensor A,
