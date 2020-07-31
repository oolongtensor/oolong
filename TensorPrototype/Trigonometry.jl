include("Tensors.jl")
include("Operations.jl")

struct SineOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    function SineOperation(A::AbstractTensor{rank}) where rank
        new{rank}((), (A,))
    end
end

function sin(A::AbstractTensor{0})
    return SineOperation(A)
end

function sin(s::Scalar)
    return sin(Tensor([s]))
end

struct CosineOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    function CosineOperation(A::AbstractTensor{rank}) where rank
        new{rank}((), (A,))
    end
end

function cos(A::AbstractTensor{0})
    return CosineOperation(A)
end

function cos(s::Scalar)
    return cos(Tensor([s]))
end

struct TangentOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    function TangentOperation(A::AbstractTensor{rank}) where rank
        new{rank}((), (A,))
    end
end

function tan(A::AbstractTensor{0})
    return TangentOperation(A)
end

function tan(s::Scalar)
    return tan(Tensor([s]))
end