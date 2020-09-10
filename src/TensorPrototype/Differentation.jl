"""Differentiates A w.r.t y"""
function _differentiate(A::Union{Tensor{T, 0}, ConstantTensor{T, 0}}, y::VariableTensor{0}, difffn) where T
    return A.value == y || A.value == [y] ? ConstantTensor(1) : ZeroTensor()
end

function _differentiate(A::VariableTensor{0}, y::VariableTensor{0}, difffn)
    return A == y ? ConstantTensor(1) : ZeroTensor()
end

function _differentiate(Z::ZeroTensor{0}, y::VariableTensor{0}, difffn)
    return Z
end

function _differentiate(add::AddOperation{0}, y::VariableTensor{0}, difffn)
    return +([difffn(child) for child in add.children]...)
end

function _differentiate(op::OuterProductOperation{0}, y::VariableTensor{0}, difffn)
    A, B = op.children
    return A*difffn(B) + difffn(A)*B
end

function _differentiate(div::DivisionOperation{0}, y::VariableTensor, difffn)
    A, B = div.children
    return (difffn(A)*B - A*difffn(B))/(B^2)
end

function _differentiate(si::SineOperation{0}, y::VariableTensor{0}, difffn)
    return Base.cos(si.children[1]) * difffn(si.children[1])
end

function _differentiate(co::CosineOperation{0}, y::VariableTensor{0}, difffn)
    return - Base.sin(co.children[1]) * difffn(co.children[1])
end

function _differentiate(ta::TangentOperation{0}, y::VariableTensor{0}, difffn)
    return difffn(ta.children[1]) / (cos(ta.children[1])^2)
end

function _differentiate(asi::ArcsineOperation{0}, y::VariableTensor{0}, difffn)
    return difffn(asi.children[1]) / sqrt(Tensor(1) - asi.children[1]^2)
end

function _differentiate(aco::ArccosineOperation{0}, y::VariableTensor{0}, difffn)
    return - difffn(aco.children[1]) / sqrt(Tensor(1) - aco.children[1]^2)
end

function _differentiate(ata::ArctangentOperation{0}, y::VariableTensor{0}, difffn)
    return difffn(ata.children[1]) / (Tensor(1) + ata.children[1]^2)
end

function _differentiate(root::RootNode, y::VariableTensor{0}, diffn)
    return RootNode(diffn(root.children[1]))
end

"""Differentiates A w.r.t y"""
function differentiate(A::AbstractTensor{0}, x::VariableTensor{0})
    return traversal(A, x -> x, _differentiate, nothing, x, nothing, true)
end

function differentiate(y::Number, x::VariableTensor{0})
    return 0
end

function _divergence(A::Tensor{T, 1}, vars::Tuple{Vararg{VariableTensor{0}}}, divergencefn) where T
    return +([Tensor(differentiate(A.value[i], vars[i])) for i in 1:length(vars)]...)
end

function _divergence(op::OuterProductOperation{1}, vars::Tuple{Vararg{VariableTensor{0}}}, divergencefn)
    x, A = op.children
    if length(A.shape) == 0
        x, A = A, x
    end
    if A isa AddOperation
        return divergencefn(+([x*child for child in A.children]...))
    elseif A isa Tensor{T,1} where T
        return divergencefn(Tensor([y*x for y in A.value], A.shape...))
    elseif A isa OuterProductOperation
        y, B = A.children
        if length(B.shape) == 0
            y, B = B, y
        end
        return divergencefn((x * y) * B)
    end
end



function _divergence(A::AddOperation{1}, vars::Tuple{Vararg{VariableTensor{0}}}, divergencefn)
    return +([divergencefn(child) for child in A.children]...)
end

function _divergence(A::RootNode, vars::Tuple{Vararg{VariableTensor{0}}}, divergencefn)
    return RootNode(divergencefn(A.children[1]))
end

function divergence(A::AbstractTensor{1}, vars::Vararg{VariableTensor{0}})
    if dim(A.shape[1]) != length(vars)
        throw(DimensionMismatch(string(vars, " does not have enough variables to gradient ", A)))
    end
    return traversal(A, x -> x, _divergence, nothing, vars, nothing, true)
end