include("Node.jl")
include("VectorSpace.jl")
include("Indices.jl")

import Base

"""A type for all the tensors and tensor like objects.

The rank means the number of indices. For example AbstractTensor{0} is a
scalar, and AbstractTensor{2} a matrix.
"""
abstract type AbstractTensor{rank} <: Node end

"""A type for the tensors which are terminal nodes, i.e not created by
operations.
"""
abstract type TerminalTensor{rank} <: AbstractTensor{rank} end

struct ScalarVariable
    name::String
end

"""A union type for everything that can be treated as a scalar."""
Scalar = Union{ScalarVariable, Base.Complex, Base.Real, AbstractTensor{0}}

"""Tensors of which we only know in which vector spaces their indices are."""
struct VariableTensor{rank} <: TerminalTensor{rank}
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{}
    freeindices::Tuple{}
    function VariableTensor(shape::Vararg{AbstractVectorSpace})
        new{length(shape)}(shape, (), ())
    end
end

function checktensordimensions(x::AbstractArray, Vs::Vararg{AbstractVectorSpace})
    if size(x) == (1,) && length(Vs) == 0
    elseif ndims(x) != length(Vs)
        throw(DomainError((x, Vs) , string(x, " does not  fit into the tensor space ", Vs)))
    else
        for i in 1:ndims(x)
            if size(x)[i] != dim(Vs[i])
                throw(DomainError(Vs[i], string("Dimension ",i,  " of ", x," does not match with vector space rank")))
            end
        end
    end
end

"""Tensor created from a multidimensional array."""
struct Tensor{T, rank} <: TerminalTensor{rank}
    value::Array{T}
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{}
    freeindices::Tuple{}
    # TODO Check x consists of scalars, if possible
    function Tensor(x::Array{T}, Vs::Vararg{AbstractVectorSpace}) where T
        checktensordimensions(x, Vs...)
        new{T, length(Vs)}(x, Vs, (), ())
    end
end

"""The general identity tensor."""
struct DeltaTensor{rank} <: TerminalTensor{rank}
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{}
    freeindices::Tuple{}
    function DeltaTensor(As::Vararg{AbstractVectorSpace})
        new{length(As)}(As, (), ())
    end
end

struct ZeroTensor{rank} <: TerminalTensor{rank}
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{}
    freeindices::Tuple{}
    function ZeroTensor(As::Vararg{AbstractVectorSpace})
        new{length(As)}(As, (), ())
    end
end

"""A tensor where every entry is of the same value."""
struct ConstantTensor{T, rank} <: TerminalTensor{rank}
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{}
    freeindices::Tuple{}
    value::T
    function ConstantTensor(value::T, As::Vararg{AbstractVectorSpace}) where (T <: Scalar)
        if value == 0
            return ZeroTensor(As)
        end 
        new{T, length(As)}(As, (), (), value)
    end
end

"""A convenience function. Allows calling Tensor on any scalar."""
Tensor(x::AbstractTensor{0}) = x
"""Turns non-tensor scalar into a tensor."""
Tensor(x::T) where (T <: Scalar) = ConstantTensor(x)

function printtensor(io, s::String, A::AbstractTensor)
    print(io, typeof(A), ", ", s, "shape: ")
    if A.shape == ()
        print(io, "()")
    else
        print(io, A.shape[1])
        for V in A.shape[2:end]
            print(io, "⊗ ",V)
        end
    end
end

Base.show(io::IO, A::Union{Tensor, ConstantTensor}) = printtensor(io, string(A.value, ", "), A)

Base.show(io::IO, A::Union{VariableTensor, DeltaTensor, ZeroTensor}) = printtensor(io, "", A)
