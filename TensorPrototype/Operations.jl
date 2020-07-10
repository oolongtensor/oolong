include("Tensors.jl")
include("Indices.jl")

import Base

abstract type Operation <: AbstractTensor end

struct AddOperation <: Operation
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{Vararg{AbstractTensor}}
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
    return AddOperation(nodes[1].shape, nodes)
end

struct IndexingOperation <: Operation
    shape::Tuple{}
    children::Tuple{AbstractTensor, Indices}
end

IndexingOperation(children::Tuple{AbstractTensor, Indices}) = IndexingOperation((), children)

function Base.getindex(x::AbstractTensor, y::Indices)
    if length(x.shape) != length(y.indices)
        error("Invalid number of indices")
    end
    for i in 1:length(y.indices)
        if x.shape[i] != y.indices[i].V
            error("Invalid vector space")
        end
    end
    return IndexingOperation((x, y))
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

struct ContractionOperation <: Operation
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{AbstractTensor, Int, Int}
end

function contr(A::AbstractTensor, i::Int, j::Int)
    if i == j
        error("Repeat index in contraction")
    end
    if A.shape[i] != dual(A.shape[j])
        error("Not contracting over dual")
    end
    if i > j
        i, j = j, i
    end
    shape = tuple(A.shape[1:(i-1)]..., A.shape[(i+1):(j-1)]..., A.shape[(j+1):length(A.shape)]...)
    return ContractionOperation(shape, (A, i, j))
end
