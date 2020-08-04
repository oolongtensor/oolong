include("Tensors.jl")
include("Operations.jl")

import Base

struct SineOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    freeindices::Tuple{}
    function SineOperation(A::AbstractTensor{rank}) where rank
        new{rank}((), (A,), ())
    end
end

function Base.sin(s::Union{ScalarVariable, AbstractTensor{0}})
    return SineOperation(Tensor(s))
end

struct CosineOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    freeindices::Tuple{}
    function CosineOperation(A::AbstractTensor{rank}) where rank
        new{rank}((), (A,), ())
    end
end

function Base.cos(s::Union{ScalarVariable, AbstractTensor{0}})
    return CosineOperation(Tensor(s))
end

struct TangentOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    freeindices::Tuple{}
    function TangentOperation(A::AbstractTensor{rank}) where rank
        new{rank}((), (A,),())
    end
end

function Base.tan(s::Union{ScalarVariable, AbstractTensor{0}})
    return TangentOperation(Tensor(s))
end