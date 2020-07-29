include("TreeVisitor.jl")

function _removezero!(add::AddOperation)
    filter!(x -> !(x isa ZeroTensor), add.children)
end

function _removezero!(node::Node)
    return
end

function _removeredundantadds!(node::Node)
    if length(node.children) == 0
        return
    end
    filter!(x -> !(x isa AddOperation && length(x.children) == 0), node.children)
    for (i, child) in enumerate(node.children)
        if child isa AddOperation && length(child.children) == 1
            node.children[i] = child.children[1]
        end
    end
end

function removezero!(node::Node)
    node = traversal(node, _removezero!, true)
    node = traversal(node, _removeredundantadds!, true)
    return node
end