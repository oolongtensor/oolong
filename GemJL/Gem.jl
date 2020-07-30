abstract type GemNode end

abstract type GemTensor <: GemNode end

struct LiteralGemTensor{T<:Scalar} <: GemTensor
    value::Array{T}
    children::Tuple{}
end

struct VariableGemTensor <: GemTensor
    shape::Tuple{Int}
    children::Tuple{}
end

struct IndexSumGem <: GemTensor
    shape::Tuple{}
    children::Tuple{GemTensor, Indices}
end

struct AddGem <: GemTensor
    shape::Tuple{Int}
    children::Tuple{Vararg{GemTensor}}
end

struct OuterProductGem <: GemTensor
    shape::Tuple{Int}
    children::Tuple{GemTensor, GemTensor}
end

struct IndexingOperation <: GemTensor
    shape::Tuple{}
    children::Tuple{AbstractTensor, Indices}
    freeindices::Tuple{Vararg{FreeIndex}}
end