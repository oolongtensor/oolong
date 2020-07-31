include("Trigonometry.jl")

function diff(A::Union{Tensor{T, 0}, ConstantTensor{T, 0}}, y::ScalarVariable) where T
    if A.value == y || A.value == [y]
        return ConstantTensor(1)
    else
        return ZeroTensor()
    end
end

function diff(Z::ZeroTensor{0}, y::ScalarVariable)
    return Z
end

function diff(add::AddOperation{0}, y::ScalarVariable)
    return +([diff(child, y) for child in add.children]...)
end

function diff(op::OuterProductOperation{0}, y::ScalarVariable)
    (A, B) = op.children
    return A*diff(B, y) + diff(A, y)*B
end

function diff(si::SineOperation{0}, y::ScalarVariable)
    return cos(si.children[1]) * diff(si.children[1], y)
end

function diff(co::CosineOperation{0}, y::ScalarVariable)
    return - sin(co.children[1]) * diff(co.children[1], y)
end