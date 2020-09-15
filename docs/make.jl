using Documenter, TensorDSL

makedocs(sitename="Oolong",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
))