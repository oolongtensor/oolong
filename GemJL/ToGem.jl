include("../TensorPrototype/Differentation.jl")
include("Gem.jl")

function toGem(A::Tensor{T}) where T
    if T <: Number
        return LiteralGemTensor{T}(A.value)
    else
        return PartiallyVariableGemTensor(A.value)
    end
end

function toGem(add::AddOperation)
    return AddGem([toGem(child) for child in add.children]...)
end