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

struct ScalarMulOperation <: Operation
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{Scalar, AbstractTensor}
end

function Base.:*(x::Scalar, A::AbstractTensor)
    return ScalarMulOperation(A.shape, (x, A))
end

function Base.:-(A::AbstractTensor)
    return -1*A
end

function Base.:-(A::AbstractTensor, B::AbstractTensor)
    return A + (-1*B)
end

struct IndexingOperation <: ScalarOperation
    children::Tuple{AbstractTensor, Indices}
end

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

#=
function Base.getindex(x::AbstractTensor, ys::Vararg{Union{String, Int, Index}})
    indexarray = []
    for i in ys
end
=#
