struct SineOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    freeindices::Tuple{}
    function SineOperation(A::AbstractTensor{rank}) where rank
        new{rank}(A.shape, (A,), ())
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
        new{rank}(A.shape, (A,), ())
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
        new{rank}(A.shape, (A,),())
    end
end

function Base.tan(s::AbstractTensor{0})
    return TangentOperation(Tensor(s))
end

struct ArcsineOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    freeindices::Tuple{}
    function ArcsineOperation(A::AbstractTensor{rank}) where rank
        if A isa ConstantTensor{T} where T<: Number && abs(A.value) > 1
            throw(DomainError(A, " asin is not defined for |x| > 1."))
        end
        new{rank}(A.shape, (A,),())
    end
end

function Base.asin(s::AbstractTensor{0})
    return ArcsineOperation(Tensor(s))
end

struct ArccosineOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    freeindices::Tuple{}
    function ArccosineOperation(A::AbstractTensor{rank}) where rank
        if A isa ConstantTensor{T} where T<: Number && abs(A.value) > 1
            throw(DomainError(A, " acos is not defined for |x| > 1."))
        end
        new{rank}(A.shape, (A,),())
    end
end

function Base.acos(s::AbstractTensor{0})
    return ArccosineOperation(Tensor(s))
end

struct ArctangentOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    freeindices::Tuple{}
    function ArctangentOperation(A::AbstractTensor{rank}) where rank
        new{rank}(A.shape, (A,),())
    end
end

function Base.atan(s::AbstractTensor{0})
    return ArctangentOperation(Tensor(s))
end