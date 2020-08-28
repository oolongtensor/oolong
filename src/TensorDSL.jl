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
const execute = PyNULL()

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
    from firedrake.parameters import parameters
    import tsfc
    gem = tsfc.fem.gem
    from gem import indices as make_indices
    impero_utils = tsfc.fem.gem.impero_utils
    import tsfc.loopy as tsfcloopy
    generate_loopy = tsfcloopy.generate
    import numpy as np
    import loopy
    from loopy.codegen import generate_code_for_a_single_kernel

    # Copy pasted from https://github.com/firedrakeproject/firedrake/blob/2d0351fa769da4fa2d807355526e9400b778fb66/firedrake/slate/slac/compiler.py#L605
    def gem_to_loopy(gem_expr):
        '''Method encapsulating stage 2.
        Converts the gem expression dag into imperoc first, and then further into loopy.
        :return slate_loopy: loopy kernel for slate operations.
        '''
        # Creation of return variables for outer loopy
        shape = gem_expr.shape if len(gem_expr.shape) != 0 else (1,)
        idx = make_indices(len(shape))
        indexed_gem_expr = gem.Indexed(gem_expr, idx)
        arg = [loopy.GlobalArg("output", shape=shape)]
        ret_vars = [gem.Indexed(gem.Variable("output", shape), idx)]

        preprocessed_gem_expr = impero_utils.preprocess_gem([indexed_gem_expr])

        # glue assignments to return variable
        assignments = list(zip(ret_vars, preprocessed_gem_expr))

        # Part A: slate to impero_c
        impero_c = impero_utils.compile_gem(assignments, (), remove_zeros=False)

        # Part B: impero_c to loopy
        return generate_loopy(impero_c, arg, parameters["form_compiler"]["scalar_type"], "slate_loopy", [])

    def execute(gem_expr):
        knl = gem_to_loopy(gem_expr)
        return loopy.generate_code_v2(knl).device_code()

    """
    copy!(gemtoloopy, py"gem_to_loopy")
    copy!(execute, py"execute")

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

loopy, gem, tsfc, isinst, impero_utils, np, generate_loopy, gemtoloopy, execute

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
