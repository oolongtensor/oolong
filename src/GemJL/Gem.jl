abstract type GemNode <: Node end

abstract type GemTensor <: GemNode end

abstract type ScalarGem <: GemNode end

abstract type GemTerminal <: GemTensor end

abstract type GemConstant <: GemTerminal end

### Indices ###

struct VariableGemIndex
    expression::Scalar
end

```Free Index
```
struct GemIndex
    extent::Int
    name::String
    id::Int
end

GemIndexTypes = Union{Int, GemIndex}

### Terminal nodes ###

struct LiteralGemTensor{T<:Number} <: GemConstant
    value::Array{T}
    children::Tuple{}
end

LiteralGemTensor(value::Array{T}) where {T<:Number} = LiteralGemTensor{T}(value, ())

struct ZeroGemTensor <: GemConstant
    shape::Tuple{Int}
    children::Tuple{}
end

ZeroGemTensor(shape::Tuple{Int}) = ZeroGemTensor(shape, ())

struct IdentityGemTensor <: GemConstant
    shape::Tuple{Int}
    children::Tuple{}
end

IdentityGemTensor(shape::Tuple{Int}) = IdentityGemTensor(shape, ())

struct VariableGemTensor <: GemTerminal
    shape::Tuple{Int}
    children::Tuple{}
end

VariableGemTensor(shape::Tuple{Int}) = VariableGemTensor(shape, ())

function shape(A::LiteralGemTensor)
    return size(A.value)
end

function shape(A::GemTensor)
    return A.shape
end

### Tensor nodes ###

struct IndexSumGem <: ScalarGem
    children::Tuple{Scalar}
    index::GemIndex
    freeindices::Tuple{Vararg{GemIndex}}
    function IndexSumGem(expr::Scalar, index::GemIndex)
        new((expr,), index, expr.freeindices)
    end
end

struct ComponentTensorGem <: GemTensor
    shape::Tuple{Int}
    children::Tuple{Scalar}
    indices::Tuple{Vararg{}}
    freeindices::Tuple{Vararg{GemIndex}}
    function ComponentTensorGem(expr::Scalar, indices::Vararg{GemIndex})
        shape = tuple([index.extent for index in indices]...)
        # TODO check for zero expression
        new(shape, expr, setdiff(expr.freeindices, indices))
    end
end

struct IndexedGem <: ScalarGem
    children::Tuple{GemTensor}
    indices::Tuple{Vararg{GemIndex}}
    freeindices::Tuple{Vararg{GemIndex}}
    function IndexedGem(expr::GemTensor, indices::Vararg{GemIndexTypes})
        if indices isa Tuple{Int} && expr isa GemConstant
            # TODO index the literal
        end
        new(expr, indices, (expr.freeindices..., [i for i in indices if i isa GemIndex]...))
    end
end

# TOODO listtensor

### Scalar operations ###

struct MathFunctionGem <: ScalarGem
    name::String
    children::Tuple{Vararg{Scalar}}
    freeindices::Tuple{Vararg{GemIndex}}
    function MathFunctionGem(name::String, expr::Scalar)
        new(name, (expr,), expr.freeindices)
    end
end

struct SumGem <: ScalarGem
    children::Tuple{Vararg{ScalarGem}}
    freeindices::Tuple{Vararg{GemIndex}}
    function SumGem(exprs::Vararg{ScalarGem})
        new(exprs, (union([exprs.freeindices for expr in exprs])...))
    end
end

struct ProductGem <: ScalarGem
    children::Tuple{ScalarGem, ScalarGem}
    freeindices::Tuple{Vararg{GemIndex}}
    function ProductGem(expr1::ScalarGem, expr2::ScalarGem)
        new((expr1, expr2), (union(expr1.freeindices, expr2.freeindices)...))
    end
end


