function execute(node::Node, variables)
    return executegem(togem(node), findgemvariables(node), variables)
end

function execute(node::Node)
    return execute(node, [])
end

struct Kernel
    knl::PyObject
    shape::Tuple{Vararg{Int}}
    variables::Array{PyObject}
    function Kernel(node::Node)
        variables = findgemvariables(node)
        gemexpr = togem(node)
        shape = gemexpr.shape
        knl = gemtoop2knl(gemexpr, variables)
        new(knl, shape, variables)
    end
end

function execute(knl::Kernel, variables::Dict)
    return executeop2knl(knl.knl, knl.shape, [variables[var.name] for var in knl.variables])
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

function findgemvariables(node::Node)
    return [togem(var) for var in findvariables(node)]
end