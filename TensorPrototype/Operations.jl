include("Tensors.jl")
include("Indices.jl")

import Base

abstract type Operation <: AbstractTensor end

struct IndexSumOperation <: Operation
    shape::Tuple{}
    children::MutableLinkedList{Node}
    freeindices::Tuple{Vararg{FreeIndex}}
    index::Int
end

IndexSumOperation(A::AbstractTensor, indices::Indices, freeindices::Vararg{FreeIndex}) = IndexSumOperation((), MutableLinkedList{Node}(A, indices), freeindices, getcounter())

#  Check if we have have an upper and lower index - if so, repeat them
function contractioncheck(A::AbstractTensor)
    contractionindices = []
    for (i, index) in enumerate(A.freeindices)
        if index' in A.freeindices[(i+1):end]
            push!(contractionindices, index)
        end
    end
    if isempty(contractionindices)
        return A
    end
    freeindices = tuple(setdiff(A.freeindices, [i for i in contractionindices], [i' for i in contractionindices])...)
    return IndexSumOperation(A, Indices(contractionindices...), freeindices...)
end

struct AddOperation <: Operation
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::MutableLinkedList{AbstractTensor}
    freeindices::Tuple{Vararg{FreeIndex}}
    index::Int
    function AddOperation(shape::Tuple{Vararg{AbstractVectorSpace}}, children::Tuple{Vararg{AbstractTensor}}, freeindices::Tuple{Vararg{FreeIndex}})
        return contractioncheck(new(shape, MutableLinkedList{AbstractTensor}(children...), freeindices, getcounter()))
    end
end

function Base.:+(nodes::Vararg{AbstractTensor})
    if length(nodes) > 1
        for node in nodes[2:length(nodes)]
            if node.shape != nodes[1].shape
                throw(DimensionMismatch(string("Shapes ", nodes[1].shape, " and ", node.shape, " don't match")))
            end
        end
    end
    return AddOperation(nodes[1].shape, nodes, tuple(union([node.freeindices for node=nodes]...)...))
end

struct OuterProductOperation <: Operation
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::MutableLinkedList{AbstractTensor}
    freeindices::Tuple{Vararg{FreeIndex}}
    index::Int
    function OuterProductOperation(shape::Tuple{Vararg{AbstractVectorSpace}}, children::Tuple{AbstractTensor, AbstractTensor}, freeindices::Tuple{Vararg{FreeIndex}})
        return contractioncheck(new(shape, MutableLinkedList{AbstractTensor}(children...), freeindices, getcounter()))
    end
end

function ⊗(x::AbstractTensor, y::AbstractTensor)
    return OuterProductOperation((x.shape..., y.shape...), (x, y),  (x.freeindices..., y.freeindices...))
end

function Base.:*(A::AbstractTensor, B::AbstractTensor)
    if !isempty(A.shape) || !isempty(B.shape)
        throw(DomainError(string(A , " or ", B, " is not a scalar")))
    end
    return A⊗B
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
    children::MutableLinkedList{Node}
    freeindices::Tuple{Vararg{FreeIndex}}
    index::Int
    function IndexingOperation(x::AbstractTensor, indices::Indices)
        return contractioncheck(new((),MutableLinkedList{Node}(x, indices), tuple(x.freeindices..., [i for i=indices.indices if i isa FreeIndex]...), getcounter()))
    end
end

struct ComponentTensorOperation <: Operation
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::MutableLinkedList{Node}
    freeindices::Tuple{Vararg{FreeIndex}}
    index::Int
end

ComponentTensorOperation(shape::Tuple{Vararg{AbstractVectorSpace}}, A::AbstractTensor, indices::Indices, freeindices::Tuple{Vararg{FreeIndex}}) = ComponentTensorOperation(shape, MutableLinkedList{Node}(A, indices), freeindices, getcounter())

function componenttensor(A::AbstractTensor, indices::Vararg{Index})
    if length(indices) == 0
        return A
    end
    if !(indices ⊆ A.freeindices)
        throw(DomainError(string("The free indices ", indices, " are not a subset of ", A.freeindices)))
    end
    shape = tuple(A.shape..., [i.V for i in indices]...)
    freeindices = tuple(setdiff(A.freeindices, indices, [i' for i in indices])...)
    return ComponentTensorOperation(shape, A, Indices(indices...), freeindices)
end

idcounter = 0

function Base.getindex(x::AbstractTensor, ys::Vararg{Index})
    if length(ys) == 0
        return x
    end
    if length(x.shape) < length(ys)
        throw(BoundsError(x.shape, ys))
    end
    for i in 1:length(ys)
        if x.shape[i] != ys[i].V
            throw(DimensionMismatch(string(ys[i], " is not in the vector space ", x.shape[i])))
        end
    end
    addedindices = []
    global idcounter
    for i in 1:(length(x.shape) - length(ys))
        push!(addedindices, FreeIndex(x.shape[length(ys) + i], "", idcounter))
        idcounter += 1
    end
    return componenttensor(IndexingOperation(x, Indices(ys..., addedindices...)), addedindices...)
end

function Base.getindex(x::AbstractTensor, ys::Vararg{Union{String, Int, Index}})
    indexarray = []
    for i in 1:length(ys)
        push!(indexarray, toindex(x.shape[i], ys[i]))
    end
    return Base.getindex(x, indexarray...)
end

function Base.show(io::IO, op::Operation, depth::Int)
    println(io, ["\t" for i in 1:depth]..., typeof(op))
    for child in op.children
        if child isa Operation
            Base.show(io, child, depth + 1)
        else
            println(io, ["\t" for i in 1:(depth+1)]..., child)
        end
    end
end

Base.show(io::IO, op::Operation) = Base.show(io, op, 0)
