abstract type AbstractVectorSpace end

struct DualVectorSpace <: AbstractVectorSpace
    dual::AbstractVectorSpace
end

mutable struct VectorSpace <: AbstractVectorSpace
    dim::Integer
    dual::DualVectorSpace
    function VectorSpace(dim::Integer)
        x = new(dim)
        x.dual = DualVectorSpace(x)
    end
end

dual(V::AbstractVectorSpace) = V.dual
