"""Differentiates A w.r.t y"""
function differentiate(A::Union{Tensor{T, 0}, ConstantTensor{T, 0}}, y::VariableTensor{0}) where T
    return A.value == y || A.value == [y] ? ConstantTensor(1) : ZeroTensor()
end

function differentiate(A::VariableTensor{0}, y::VariableTensor{0})
    return A == y ? ConstantTensor(1) : ZeroTensor()
end

"""Differentiates A w.r.t y"""
function differentiate(Z::ZeroTensor{0}, y::VariableTensor{0})
    return Z
end

"""Differentiates A w.r.t y"""
function differentiate(add::AddOperation{0}, y::VariableTensor{0})
    return +([differentiate(child, y) for child in add.children]...)
end

"""Differentiates A w.r.t y"""
function differentiate(op::OuterProductOperation{0}, y::VariableTensor{0})
    (A, B) = op.children
    return A*differentiate(B, y) + differentiate(A, y)*B
end

"""Differentiates A w.r.t y"""
function differentiate(si::SineOperation{0}, y::VariableTensor{0})
    return Base.cos(si.children[1]) * differentiate(si.children[1], y)
end

"""Differentiates A w.r.t y"""
function differentiate(co::CosineOperation{0}, y::VariableTensor{0})
    return - Base.sin(co.children[1]) * differentiate(co.children[1], y)
end