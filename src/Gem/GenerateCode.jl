function createop2knl(node::Node)
    gem_expr = togem(node)
    return gemtoop2knl(gem_expr, [togem(var) for var in findvariables(node)])
end

function execute(node::Node)
    return executegem(togem(node))
end

function _findvariables(tensor::VariableTensor)
    return Set{VariableTensor}([tensor])
end

function _findvariables(node::Node)
    return Set{VariableTensor}()
end

function _findvariables(root::RootNode, found::Vararg{Set{VariableTensor}})
    return RootNode(union(found...))
end

function _findvariables(node::Node, found::Vararg{Set{VariableTensor}})
    return union(found...)
end

function findvariables(node::Node)
    return traversal(node, x->x, _findvariables, nothing, nothing)
end