struct SineOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    freeindices::Tuple{}
    function SineOperation(A::AbstractTensor{rank}) where rank
        new{rank}((), (A,), ())
    end
end

function Base.sin(s:: AbstractTensor{0})
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

function Base.cos(s::AbstractTensor{0})
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

function Base.tan(s::AbstractTensor{0})
    return TangentOperation(Tensor(s))
end