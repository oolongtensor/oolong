__precompile__()
module TensorDSL

using PyCall

const gem = PyNULL()
const tsfc = PyNULL()
const isinst = PyNULL()
const impero_utils = PyNULL()
const loopy = PyNULL()
const np = PyNULL()
const generate_loopy = PyNULL()
const gemtoloopy = PyNULL()

# https://github.com/JuliaPy/PyCall.jl/blob/master/README.md#using-pycall-from-julia-modules
function __init__()
    copy!(tsfc, pyimport("tsfc"))
    copy!(gem, tsfc.fem.gem)
    copy!(isinst, pybuiltin("isinstance"))
    copy!(impero_utils, gem.impero_utils)
    copy!(loopy, pyimport("loopy"))
    copy!(np, pyimport("numpy"))
    copy!(generate_loopy, pyimport("tsfc.loopy").generate)
    py"""
    import tsfc
    gem = tsfc.fem.gem
    impero_utils = tsfc.fem.gem.impero_utils
    import tsfc.loopy as tsfcloopy
    generate_loopy = tsfcloopy.generate
    import numpy as np
    import loopy

    def createreturnvalue(gem_expr):
        indices = tuple(gem.Index(n) for n in gem_expr.shape) 
        free_shape = tuple(i.extent for i in gem_expr.free_indices)
        return gem.Indexed(gem.Variable('R', gem_expr.shape + free_shape),
            indices + gem_expr.free_indices), indices

    def gemtoloopy(gem_expr):
        return_value, indices = createreturnvalue(gem_expr)
        outer_indices = indices + gem_expr.free_indices
        arg = loopy.GlobalArg('R', dtype=np.float64, shape=tuple(i.extent for i in outer_indices))
        imperoc = impero_utils.compile_gem([(return_value, gem.Indexed(gem_expr, indices))], outer_indices)
        return generate_loopy(imperoc, [arg], np.dtype(np.float64))
    """
    copy!(gemtoloopy, py"gemtoloopy")
end

export
Node,

AbstractVectorSpace, VectorSpace, DualVectorSpace, RnSpace, dual, dim,

Index, FreeIndex, FixedIndex, Indices, toindex,

AbstractTensor, TerminalTensor, ScalarVariable, Scalar, VariableTensor,
Tensor, DeltaTensor, ZeroTensor, ConstantTensor,

Operation, IndexSumOperation, AddOperation, OuterProductOperation, ⊗,
IndexingOperation, ComponentTensorOperation, componenttensor,

SineOperation, sin, CosineOperation, cos, TangentOperation, tan,

diff,

Assignment, assign, RootNode,

updatechildren, updatevectorspace,

togem, toloopy,

loopy, gem, tsfc, isinst, impero_utils, np, generate_loopy, gemtoloopy

TensorDSL
include("TensorPrototype/Node.jl")
include("TensorPrototype/VectorSpace.jl")
include("TensorPrototype/Indices.jl")
include("TensorPrototype/Tensors.jl")
include("TensorPrototype/Operations.jl")
include("TensorPrototype/Trigonometry.jl")
include("TensorPrototype/Differentation.jl")
include("TreeVisitor/Traversal.jl")
include("TreeVisitor/UpdateNodes.jl")
include("TensorPrototype/Assignment.jl")
include("Gem/togem.jl")
include("Gem/GenerateCode.jl")

end
