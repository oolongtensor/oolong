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

struct IndexingOperation <: Operation
    shape::Tuple{}
    children::Tuple{AbstractTensor, Indices}
    freeindices::Set{FreeIndex}
end

IndexingOperation(x::AbstractTensor, indices::Indices) = IndexingOperation((),(x, indices), union(Set([i for i=indices.indices if i isa FreeIndex]), x.freeindices))

struct ComponentTensorOperation <: Operation
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{AbstractTensor, Indices}
    freeindices::Set{FreeIndex}
end

function componentTensor(A::AbstractTensor, indices::Vararg{Index})
    if !(indices ⊆ A.freeindices)
        error("Not all indices are free indices")
    end
    shape = tuple(A.shape..., [i.V for i in indices]...)
    freeindices = setdiff(A.freeindices, indices)
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
    if length(ys) < length(x.shape)
        addedindices = []
        global idcounter
        for i in 1:(length(x.shape) - length(ys))
            push!(addedindices, FreeIndex(x.shape[length(ys) + 1], "", idcounter))
            idcounter += 1
        end
        op = IndexingOperation(x, Indices(ys..., addedindices...))
        for i in 1:(length(x.shape) - length(ys))
            op = componentTensor(op, addedindices[i])
        end
        return op
    else
        return IndexingOperation(x, Indices(ys...))
    end
end


function Base.getindex(x::AbstractTensor, ys::Vararg{Union{String, Int, Index}})
    indexarray = []
    for i in 1:length(ys)
        push!(indexarray, toindex(x.shape[i], ys[i]))
    end
    return Base.getindex(x, indexarray...)
end

IndexSumIndex(is::Vararg{Int}) = IndexSumIndex(is, ())

struct IndexSumOperation <: Operation
    shape::Tuple{}
    children::Tuple{AbstractTensor, Indices}
    freeindices::Set{FreeIndex}
end

IndexSumOperation(A::AbstractTensor, i::Index, freeindices::Set{FreeIndex}) = IndexSumOperation((), (A, Indices(i)), freeindices)

# Sums over i:s and i':s
function indexsum(A::AbstractTensor, i::FreeIndex)
    if A.shape != ()
        error("Requires scalar") # TODO does it?
    end
    freeindices = setdiff(A.freeindices, [i, i'])
    return IndexSumOperation(A, i, freeindices)
end

function tensorcontraction(A::IndexingOperation, B::IndexingOperation)
    tocontract = []
    # Find the indices to contract over
    Aindices, Bindices = A.children[2].indices, B.children[2].indices
    for i in 1:min(length(Aindices), length(Bindices))
        if Aindices[end + 1 - i] == Bindices[i]'
            push!(tocontract, Aindices[end + 1 - i])
        else
            break
        end
    end
    op = A⊗B
    if length(tocontract) == 0
        return op
    end
    # Sum over the contraction indices
    for index in tocontract
        op = indexsum(op, index)
    end
    # Create the component tensor of the remaining indices
    for index in (Aindices[1:(end - length(tocontract))]..., Bindices[(length(tocontract)+1):end]...)
        op = componentTensor(op, index)
    end
    return op
end

function Base.:*(A::IndexingOperation, B::IndexingOperation)
    return tensorcontraction(A, B)
end
