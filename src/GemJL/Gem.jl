abstract type GemNode <: Node end

abstract type GemTensor <: GemNode end

abstract type ScalarExprGem <: GemNode end

abstract type GemTerminal{rank} <: GemTensor end

abstract type GemConstant{rank} <: GemTerminal{rank} end

### Terminal nodes ###

struct LiteralGemTensor{T<:Number, rank} <: GemConstant{rank} 
    value::Array{T}
    children::Tuple{}
    freeindices::Tuple{}
end

rank(a::AbstractArray) =  ndims(a) > 1 ? ndims(a) : (length(a) == 1 ? 0 : 1)

LiteralGemTensor(value::Array{T}) where T<:Number = LiteralGemTensor{T, rank(value)}(value, (), ())
LiteralGemTensor(value::T) where T<:Number = LiteralGemTensor(fill(value, ()))

struct ZeroGemTensor{rank} <: GemConstant{rank}
    shape::Tuple{Int}
    children::Tuple{}
    freeindices::Tuple{}
end

ZeroGemTensor(shape::Tuple{Int}) = ZeroGemTensor{length(shape)}(shape, (), ())

struct IdentityGemTensor{rank} <: GemConstant{rank}
    shape::Tuple{Int}
    children::Tuple{}
    freeindices::Tuple{}
end

IdentityGemTensor(shape::Tuple{Int}) = IdentityGemTensor{length(shape)}(shape, (), ())

struct VariableGemTensor{rank} <: GemTerminal{rank}
    shape::Tuple{Int}
    children::Tuple{}
    freeindices::Tuple{}
end

VariableGemTensor(shape::Tuple{Int}) = VariableGemTensor{length(shape)}(shape, (), ())

function shape(A::LiteralGemTensor)
    return size(A.value)
end

function shape(A::GemTensor)
    return A.shape
end

ScalarGem = Union{Scalar, GemTerminal{0}}

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

### Tensor nodes ###

struct IndexSumGem <: ScalarExprGem
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

struct IndexedGem <: ScalarExprGem
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

struct MathFunctionGem <: ScalarExprGem
    name::String
    children::Tuple{Vararg{Scalar}}
    freeindices::Tuple{Vararg{GemIndex}}
    function MathFunctionGem(name::String, expr::Scalar)
        new(name, (expr,), expr.freeindices)
    end
end

struct SumGem <: ScalarExprGem
    children::Tuple{Vararg{ScalarGem}}
    freeindices::Tuple{Vararg{GemIndex}}
    function SumGem(exprs::Vararg{ScalarGem})
        constants = filter!(x -> x isa GemConstant, [exprs...])
        literal = LiteralGemTensor(sum([i isa LiteralGemTensor ?
            i.value[1] : 1  for i in constants if !(i isa ZeroGemTensor)]))
        if length(constants) >= length(exprs)
            return literal
        end
        nonconstants = filter!(x -> !(x isa GemConstant), [exprs...])
        new(tuple(literal, nonconstants...), tuple(union([exprs.freeindices for expr in nonconstants])...))
    end
end

struct ProductGem <: ScalarExprGem
    children::Tuple{ScalarGem, ScalarGem}
    freeindices::Tuple{Vararg{GemIndex}}
    function ProductGem(expr1::ScalarGem, expr2::ScalarGem)
        new((expr1, expr2), (union(expr1.freeindices, expr2.freeindices)...))
    end
end


