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
    
    def gemtoloopy(gem_expr):
        return_value = gem.Variable("R", gem_expr.shape)
        arg = loopy.GlobalArg('R', dtype=np.float64, shape=(6,))

        imperoc = impero_utils.compile_gem([(return_value, gem_expr)], gem_expr.free_indices)
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
