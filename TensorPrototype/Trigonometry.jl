include("Tensors.jl")
include("Operations.jl")

struct SineOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    function SineOperation(A::AbstractTensor{rank}) where rank
        new{rank}((), (A,))
    end
end

function sin(s::Scalar)
    return SineOperation(Tensor([s]))
end

struct CosineOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    function CosineOperation(A::AbstractTensor{rank}) where rank
        new{rank}((), (A,))
    end
end

function cos(s::Scalar)
    return CosineOperation(Tensor([s]))
end

struct TangentOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    function TangentOperation(A::AbstractTensor{rank}) where rank
        new{rank}((), (A,))
    end
end

function tan(s::Scalar)
    return TangentOperation(Tensor([s]))
end