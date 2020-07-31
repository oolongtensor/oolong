include("Tensors.jl")
include("Indices.jl")

import Base

abstract type Operation{rank} <: AbstractTensor{rank} end

struct IndexSumOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor, Indices}
    freeindices::Tuple{Vararg{FreeIndex}}
end

IndexSumOperation(A::AbstractTensor, indices::Indices, freeindices::Vararg{FreeIndex}) = IndexSumOperation{0}((), (A, indices), freeindices)

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

struct AddOperation{rank} <: Operation{rank}
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{Vararg{AbstractTensor}}
    freeindices::Tuple{Vararg{FreeIndex}}
    function AddOperation(children::Tuple{Vararg{AbstractTensor{rank}}}, freeindices::Tuple{Vararg{FreeIndex}}) where rank
        if length(children) > 1
            for child in children[2:length(children)]
                if child.shape != children[1].shape
                    throw(DimensionMismatch(string("Shapes ", children[1].shape, " and ", child.shape, " don't match")))
                end
            end
        end
        newchildren = tuple(filter!(x -> !(x isa ZeroTensor), [children...])...)
        if length(newchildren) == 1
            return newchildren[1]
        end
        if length(newchildren) == 0
            return children[1]
        end
        return contractioncheck(new{rank}(children[1].shape, newchildren, freeindices))
    end
end

function Base.:+(nodes::Vararg{Node})
    return AddOperation(nodes, tuple(union([node.freeindices for node=nodes]...)...))
end

struct OuterProductOperation{rank} <: Operation{rank}
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{AbstractTensor, AbstractTensor}
    freeindices::Tuple{Vararg{FreeIndex}}
    function OuterProductOperation(shape::Tuple{Vararg{AbstractVectorSpace}}, children::Tuple{AbstractTensor, AbstractTensor}, freeindices::Tuple{Vararg{FreeIndex}})
        return contractioncheck(new{length(shape)}(shape, children, freeindices))
    end
end

function ⊗(x::AbstractTensor, y::AbstractTensor)
    return OuterProductOperation((x.shape..., y.shape...), (x, y),  (x.freeindices..., y.freeindices...))
end

function Base.:*(x::ScalarVariable, y::Scalar)
    if !(x isa AbstractTensor)
        x = Tensor([x])
    end
    if !(y isa AbstractTensor)
        y = Tensor([y])
    end
    return x ⊗ y
end

function Base.:*(x::Scalar, y::AbstractTensor)
    return Tensor(x) ⊗ A
end

function Base.:*(x::Scalar, y::ScalarVariable)
    return Tensor(x) ⊗ Tensor(y)
end

function Base.:-(A::AbstractTensor)
    return -1*A
end

function Base.:-(A::AbstractTensor{rank}, B::AbstractTensor{rank}) where rank
    return A + (-1*B)
end

struct IndexingOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor, Indices}
    freeindices::Tuple{Vararg{FreeIndex}}
    function IndexingOperation(x::AbstractTensor, indices::Indices)
        return contractioncheck(new{0}((),(x, indices), tuple(x.freeindices..., [i for i=indices.indices if i isa FreeIndex]...)))
    end
end

struct ComponentTensorOperation{rank} <: Operation{rank}
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{AbstractTensor, Indices}
    freeindices::Tuple{Vararg{FreeIndex}}
    function ComponentTensorOperation(shape::Tuple{Vararg{AbstractVectorSpace}}, children::Tuple{AbstractTensor, Indices}, freeindices::Tuple{Vararg{FreeIndex}})
        new{length(shape)}(shape, children, freeindices)
    end
end

function componenttensor(A::AbstractTensor, indices::Vararg{Index})
    if length(indices) == 0
        return A
    end
    if !(indices ⊆ A.freeindices)
        throw(DomainError(string("The free indices ", indices, " are not a subset of ", A.freeindices)))
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