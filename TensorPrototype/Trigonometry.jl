include("Tensors.jl")
include("Operations.jl")

struct SineOperation <: Operation
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    function SineOperation(A::AbstractTensor)
        if !isempty(A.shape)
            throw(DomainError(string(A, " is not a scalar")))
        end
        new((), (A,))
    end
end

function sin(A::AbstractTensor)
    return SineOperation(A)
end

function sin(s::Scalar)
    return sin(Tensor([s]))
end

struct CosineOperation <: Operation
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    function CosineOperation(A::AbstractTensor)
        if !isempty(A.shape)
            throw(DomainError(string(A, " is not a scalar")))
        end
        new((), (A,))
    end
end

function cos(A::AbstractTensor)
    return CosineOperation(A)
end

function cos(s::Scalar)
    return cos(Tensor([s]))
end

struct TangentOperation <: Operation
    shape::Tuple{}
    children::Tuple{AbstractTensor}
    function TangentOperation(A::AbstractTensor)
        if !isempty(A.shape)
            throw(DomainError(string(A, " is not a scalar")))
        end
        new((), (A,))
    end
end

function tan(A::AbstractTensor)
    return TangentOperation(A)
end

function tan(s::Scalar)
    return tan(Tensor([s]))
end