include("Node.jl")

abstract type AbstractTensor <: Node end

struct ScalarVariable <: AbstractTensor end

Scalar = Union{ScalarVariable, Base.Complex, Base.Real}

struct ConcreteTensor <: AbstractTensor
    # indices

    # Some kind of a collection of Scalars
    value
    shape
    children
end

ConcreteTensor(x::Vararg{Scalar}) = ConcreteTensor(x, (length(x),), ())
