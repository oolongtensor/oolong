abstract type GemNode end

abstract type GemTensor <: GemNode end

struct LiteralGemTensor{T<:Number} <: GemTensor
    value::Array{T}
    children::Tuple{}
end

LiteralGemTensor{T}(value::Array{T}) where (T<:Number) = LiteralGemTensor{T}(value, ())

struct PartiallyVariableGemTensor <: GemTensor
    value::Array
    children::Tuple{}
end

PartiallyVariableGemTensor(value::AbstractArray) = PartiallyVariableGemTensor(value, ())

function shape(A::Union{LiteralGemTensor, PartiallyVariableGemTensor})
    return size(A.value)
end

function shape(A::GemTensor)
    return A.shape
end

struct VariableGemTensor <: GemTensor
    shape::Tuple{Int}
    children::Tuple{}
end

struct VariableGemIndex
    # -1 for unknown
    extent::Int
    id::Int
    name::String
end

struct GemVariable
    id::Int
    name::String
end

GemIndex = Union{Int, VariableGemIndex}

struct IndexSumGem <: GemTensor
    shape::Tuple{}
    children::Tuple{GemTensor, Vararg{GemIndex}}
end

struct AddGem <: GemTensor
    shape::Tuple{Int}
    children::Tuple{Vararg{GemTensor}}
    function AddGem(nodes::Vararg{GemTensor})
        literals = tuple(filter!(x -> x isa LiteralGemTensor, [nodes...])...)
        if isempty(literals)
            new(shape(nodes[1]), nodes)
        end
        nonliterals = tuple(filter!(x -> !(x isa LiteralGemTensor), [nodes...])...)
        literal = LiteralGemTensor(+([lit.value for lit in literals]...))
        new(shape(nodes[1]), (literal, nonliterals...))
    end
end

struct OuterProductGem <: GemTensor
    shape::Tuple{Int}
    children::Tuple{GemTensor, GemTensor}
end

struct IndexingGem <: GemTensor
    shape::Tuple{Int}
    children::Tuple{AbstractTensor, Indices}
    freeindices::Tuple{Vararg{FreeIndex}}
end

struct ComponentTensorGem <: GemTensor
    shape::Tuple{Int}
    children::Tuple{AbstractTensor, Indices}
    freeindices::Tuple{Vararg{FreeIndex}}
end