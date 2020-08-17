abstract type GemNode <: Node end

abstract type GemTensor{rank} <: GemNode end

abstract type ScalarExprGem <: GemNode end

abstract type GemTerminal{rank} <: GemTensor{rank} end

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
    shape::Tuple{Vararg{Int}}
    children::Tuple{}
    freeindices::Tuple{}
end

ZeroGemTensor(shape::Vararg{Int}) = ZeroGemTensor{length(shape)}(shape, (), ())

struct IdentityGemTensor{rank} <: GemConstant{rank}
    shape::Tuple{Vararg{Int}}
    children::Tuple{}
    freeindices::Tuple{}
end

IdentityGemTensor(shape::Vararg{Int}) = IdentityGemTensor{length(shape)}(shape, (), ())

struct VariableGemTensor{rank} <: GemTerminal{rank}
    shape::Tuple{Vararg{Int}}
    children::Tuple{}
    freeindices::Tuple{}
end

VariableGemTensor(shape::Tuple{Vararg{Int}}) = VariableGemTensor{length(shape)}(shape, (), ())

function shape(A::LiteralGemTensor)
    return size(A.value)
end

function shape(A::GemTensor)
    return A.shape
end

ScalarGem = Union{ScalarExprGem, GemTerminal{0}}

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

struct ComponentTensorGem{rank} <: GemTensor{rank}
    shape::Tuple{Int}
    children::Tuple{Scalar}
    indices::Tuple{Vararg{}}
    freeindices::Tuple{Vararg{GemIndex}}
    function ComponentTensorGem(expr::Scalar, indices::Vararg{GemIndex})
        shape = tuple([index.extent for index in indices]...)
        # TODO check for zero expression
        new{length(shape)}(shape, expr, setdiff(expr.freeindices, indices))
    end
end

struct IndexedGem <: ScalarExprGem
    children::Tuple{GemTensor}
    indices::Tuple{Vararg{GemIndexTypes}}
    freeindices::Tuple{Vararg{GemIndex}}
    function IndexedGem(expr::GemTensor, indices::Vararg{GemIndexTypes})
        if indices isa Tuple{Vararg{Int}} && expr isa GemConstant
            if expr isa ZeroGemTensor
                return ZeroGemTensor()
            elseif expr isa IdentityGemTensor
                if all(indices[i] == indices[i+1] for i in 1:(length(indices)-1))
                    return LiteralGemTensor(fill(1, ()))
                else
                    return ZeroGemTensor()
                end
            elseif expr isa LiteralGemTensor
                return LiteralGemTensor(fill(expr.value[indices...], ()))
            end
        end
        new((expr,), indices, (expr.freeindices..., [i for i in indices if i isa GemIndex]...))
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
        if length(constants) == length(exprs)
            return literal
        end
        nonconstants = filter!(x -> !(x isa GemConstant), [exprs...])
        if all(literal.value .== 0)
            new(tuple(nonconstants...), (union([expr.freeindices for expr in nonconstants])...))
        else
            new(tuple(literal, nonconstants...), (union([expr.freeindices for expr in nonconstants])...))
        end
    end
end

struct ProductGem <: ScalarExprGem
    children::Tuple{ScalarGem, ScalarGem}
    freeindices::Tuple{Vararg{GemIndex}}
    function ProductGem(expr1::ScalarGem, expr2::ScalarGem)
        new((expr1, expr2), (union(expr1.freeindices, expr2.freeindices)...))
    end
end


