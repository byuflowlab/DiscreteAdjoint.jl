using DiscreteAdjoint
using Documenter

DocMeta.setdocmeta!(DiscreteAdjoint, :DocTestSetup, :(using DiscreteAdjoint); recursive=true)

makedocs(;
    modules=[DiscreteAdjoint],
    authors="Taylor McDonnell <taylor.golden.mcdonnell@gmail.com> and contributors",
    repo="https://github.com/byuflowlab/DiscreteAdjoint.jl/blob/{commit}{path}#{line}",
    sitename="DiscreteAdjoint.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://byuflowlab.github.io/DiscreteAdjoint.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Usage" => "guide.md",
        "Theory" => "theory.md",
    ],
)

deploydocs(;
    repo="github.com/byuflowlab/DiscreteAdjoint.jl",
    devbranch="main",
)
