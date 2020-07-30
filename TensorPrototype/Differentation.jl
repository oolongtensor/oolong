include("Operations.jl")

struct DifferentationOperation <: Operation
    shape::Tuple{}
    children::Tuple{AbstractTensor, ScalarVariable}
    function DifferentationOperation(A::AbstractTensor, v::ScalarVariable)
        if !isempty(A.shape)
            throw(DomainError(string(A, " is not a scalar")))
        end
        new((), (A, v))
    end
end

function diff(A::AbstractTensor, v::ScalarVariable)
    return DifferentationOperation(A, v)
end

function diff(s::Scalar, v::ScalarVariable)
    return DifferentationOperation(Tensor[s], v)
end