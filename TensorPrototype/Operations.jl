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
    return OuterProductOperation(tuple(x.shape..., y.shape...), (x, y),  union(x.freeindices, y.freeindices))
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

struct ComponentTensorIndex <: Node
    index::FreeIndex
    loc::Int # The index of the new dimension
    children::Tuple{}
end

ComponentTensorIndex(index::FreeIndex, loc::Int) = ComponentTensorIndex(index, loc, ())

struct ComponentTensorOperation <: Operation
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{AbstractTensor, ComponentTensorIndex}
    freeindices::Set{FreeIndex}
end

function componentTensor(A::AbstractTensor, i::FreeIndex, loc::Int)
    if !(i in A.freeindices)
        error("The free index to loop over is not there")
    end
    shape = tuple(A.shape[1:(loc-1)]..., i.V, A.shape[loc:length(A.shape)]...)
    ctindex = ComponentTensorIndex(i, loc)
    freeindices = setdiff(A.freeindices, [i])
    return ComponentTensorOperation(shape, (A, ctindex), freeindices)
end

# Defaults to looping over the new index at the end
function componentTensor(A::AbstractTensor, i::FreeIndex)
    return componentTensor(A, i, length(A.shape) + 1)
end
