"""
    AbstractTensor{rank}

A type for all the tensors and tensor like objects.

The rank means the number of vector spaces in the shape. For example AbstractTensor{0} is a
scalar, and AbstractTensor{2} a matrix.
"""
abstract type AbstractTensor{rank} <: Node end

"""
    TerminalTensor{rank}

A type for the tensors which are terminal nodes, i.e not created by
operations.
"""
abstract type TerminalTensor{rank} <: AbstractTensor{rank} end

"""A union type for everything that can be treated as a scalar. Includes
tensors of rank 0 and numbers."""
Scalar = Union{Number, AbstractTensor{0}}

"""
    VariableTensor(name::String, shape::Vararg{AbstractVectorSpace})

Tensors of which we only know in which vector spaces their indices are. The
rank is the lenght of the shape."""
struct VariableTensor{rank} <: TerminalTensor{rank}
    name::String
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{}
    freeindices::Tuple{}
    function VariableTensor(name::String, shape::Vararg{AbstractVectorSpace})
        new{length(shape)}(name, shape, (), ())
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

"""
    Tensor(x::Array{T}, Vs::Vararg{AbstractVectorSpace}) where T

Tensor{T, length(VS)} created from a multidimensional array."""
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

"""
    DeltaTensor(Vs::Vararg{AbstractVectorSpace})

Symbolic delta tensor. Each vector space in Vs must be either equal to other
vector spaces or their duals.
"""
struct DeltaTensor{rank} <: TerminalTensor{rank}
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{}
    freeindices::Tuple{}
    function DeltaTensor(Vs::Vararg{AbstractVectorSpace})
        if length(Vs) == 1
            throw(DomainError(Vs, "Cannot have delta of dimension 1"))
        end
        for V in Vs[2:end]
            if V != Vs[1] && V != Vs[1]'
                throw(DomainError(Vs,
                "The shape of a delta can contain only a single vector space and its dual."))
            end
        end
        new{length(Vs)}(Vs)
    end
end

"""
    ZeroTensor(Vs::Vararg{AbstractVectorSpace})

Symbolic zero tensor of the shape Vs."""
struct ZeroTensor{rank} <: TerminalTensor{rank}
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{}
    freeindices::Tuple{}
    function ZeroTensor(As::Vararg{AbstractVectorSpace})
        new{length(As)}(As, (), ())
    end
end

"""
    ConstantTensor(value::T, Vs::Vararg{AbstractVectorSpace}) where (T <: Scalar)

A tensor where every entry is of the same value. Creates
ConstantTensor{T, length(Vs)}. If value == 0, creates a [`ZeroTensor`](@ref)."""
struct ConstantTensor{T, rank} <: TerminalTensor{rank}
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{}
    freeindices::Tuple{}
    value::T
    function ConstantTensor(value::T, As::Vararg{AbstractVectorSpace}) where (T <: Scalar)
        if value == 0
            return ZeroTensor(As...)
        end
        new{T, length(As)}(As, (), (), value)
    end
end

Tensor(x::AbstractTensor{0}) = x
ConstantTensor(x::AbstractTensor{0}) = x
"""
    Tensor(x::T) where (T <: Scalar)

Turns a non-tensor scalar into a tensor. A tensor scalar does not change."""
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

Base.show(io::IO, A::VariableTensor) = printtensor(io, string(A.name, ", "), A)

Base.show(io::IO, A::Union{DeltaTensor, ZeroTensor}) = printtensor(io, "", A)
