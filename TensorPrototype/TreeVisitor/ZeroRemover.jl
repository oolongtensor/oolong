function removezero!(add::AddOperation)
    filter!(x -> !(x isa ZeroTensor), add.children)
end

function removezero!(node::Node)
    return
end