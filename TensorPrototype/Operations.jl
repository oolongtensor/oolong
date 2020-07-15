include("Tensors.jl")
include("Indices.jl")

import Base

abstract type Operation <: AbstractTensor end

struct AddOperation <: Operation
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{Vararg{AbstractTensor}}
    freeindices::Tuple{Vararg{FreeIndex}}
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
    return AddOperation(nodes[1].shape, nodes, tuple(union([node.freeindices for node=nodes]...)...))
end

struct OuterProductOperation <: Operation
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{AbstractTensor, AbstractTensor}
    freeindices::Tuple{Vararg{FreeIndex}}
end

function ⊗(x::AbstractTensor, y::AbstractTensor)
    return OuterProductOperation(tuple(x.shape..., y.shape...), (x, y),  tuple(union(x.freeindices, y.freeindices)...))
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

struct IndexingOperation <: Operation
    shape::Tuple{}
    children::Tuple{AbstractTensor, Indices}
    freeindices::Tuple{Vararg{FreeIndex}}
end

IndexingOperation(x::AbstractTensor, indices::Indices) = IndexingOperation((),(x, indices), tuple(union(x.freeindices, [i for i=indices.indices if i isa FreeIndex])...))

struct ComponentTensorOperation <: Operation
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{AbstractTensor, Indices}
    freeindices::Tuple{Vararg{FreeIndex}}
end

function componentTensor(A::AbstractTensor, indices::Vararg{Index})
    if length(indices) == 0
        return A
    end
    if !(indices ⊆ A.freeindices)
        error("Not all indices are free indices")
    end
    shape = tuple(A.shape..., [i.V for i in indices]...)
    freeindices = tuple(setdiff(A.freeindices, indices, [i' for i in indices])...)
    return ComponentTensorOperation(shape, (A, Indices(indices...)), freeindices)
end

idcounter = 0

function Base.getindex(x::AbstractTensor, ys::Vararg{Index})
    if length(ys) == 0
        return x
    end
    if length(x.shape) < length(ys)
        error("Too many indices")
    end
    for i in 1:length(ys)
        if x.shape[i] != ys[i].V
            error("Invalid vector space")
        end
    end
    addedindices = []
    global idcounter
    for i in 1:(length(x.shape) - length(ys))
        push!(addedindices, FreeIndex(x.shape[length(ys) + i], "", idcounter))
        idcounter += 1
    end
    return componentTensor(IndexingOperation(x, Indices(ys..., addedindices...)), addedindices...)
end


function Base.getindex(x::AbstractTensor, ys::Vararg{Union{String, Int, Index}})
    indexarray = []
    for i in 1:length(ys)
        push!(indexarray, toindex(x.shape[i], ys[i]))
    end
    return Base.getindex(x, indexarray...)
end

struct IndexSumOperation <: Operation
    shape::Tuple{}
    children::Tuple{IndexingOperation, Indices}
    freeindices::Tuple{Vararg{FreeIndex}}
end

IndexSumOperation(A::IndexingOperation, indices::Indices, freeindices::Tuple{FreeIndex}) = IndexSumOperation((), (A, indices), freeindices)

# Sums over the indices in the given order
function indexsum(A::IndexingOperation, indices::Vararg{FreeIndex})
    freeindices = tuple(setdiff(A.freeindices, [i for i in indices], [i' for i in indices])...)
    return IndexSumOperation(A, Indices(indices...), freeindices)
end


# TODO Is the recursive design a good idea?
function getadjacentindices(indices::Vararg{FreeIndex})
    for i in 2:length(indices)
        if indices[i]' == indices[i-1]
            return (indices[i-1], getadjacentindices(indices[1:(i-2)]..., indices[(i+1):end]...)...)
        end
    end
    return ()
end


function tensorcontraction(A::IndexingOperation)
    contractions = getadjacentindices(A.children[2].indices...)
    remaining = tuple(setdiff(A.children[2].indices, contractions, [i' for i in contractions])...)
    return componentTensor(indexsum(A, contractions...), remaining...)
end

function Base.:*(A::IndexingOperation, B::IndexingOperation)
    return tensorcontraction(A⊗B)
end
