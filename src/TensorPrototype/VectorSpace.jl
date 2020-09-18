"Abstract vector space type."
abstract type AbstractVectorSpace end

counter = 0

"""
    VectorSpace(dim::Union{Int, Nothing})

A vector space with optional dimension. Attribute id distinguishes between
different vector spaces of the same dimension. User should not need to set id.
"""
struct VectorSpace <: AbstractVectorSpace
    dim::Union{Int, Nothing}
    id::Int
    function VectorSpace(dim::Union{Int, Nothing})
        global counter
        counter += 1
        new(dim, counter)
    end
end

VectorSpace() = VectorSpace(nothing)

"""
    DualVectorSpace(vectorspace::AbstractVectorSpace)

Dual of a given vector space.
"""
struct DualVectorSpace <: AbstractVectorSpace
    vectorspace::AbstractVectorSpace
end

"""
    RnSpace(dim::Int)

Vector space corresponding to R^n for some n.
"""
struct RnSpace <: AbstractVectorSpace
    dim::Int
end

"""
    dual(V::AbstractVectorSpace)

Returns the dual of V.
"""
dual(V::VectorSpace) = DualVectorSpace(V)
dual(Vstar::DualVectorSpace) = Vstar.vectorspace
dual(R::RnSpace) = R
"""
    dim(V::AbstractVectorSpace)

Returns the dimension V.
"""
dim(V::VectorSpace) = V.dim
dim(Vstar::DualVectorSpace) = Vstar.vectorspace.dim
dim(R::RnSpace) = R.dim

"""
    Base.adjoint(V::AbstractVectorSpace)

Returns the [`dual(V)`](@ref).
"""
Base.adjoint(V::AbstractVectorSpace) = dual(V)

Base.show(io::IO, Vstar::DualVectorSpace) = print(io, Vstar', "*")

Base.show(io::IO, R::RnSpace) = print(io,  "R^",dim(R))

function Base.show(io::IO, V::VectorSpace)
    if dim(V) === nothing
        print(io, "V", "_", V.id)
    else
        print(io, "V", dim(V), "_", V.id)
    end
end
