"""
    atomSimulateElectricField(x, y, z, hRanges, kRanges, lRanges)

Simulate the electric field for a group of atoms (`x`, `y`, and `z`) on
a sequence of grids in reciprocal space (`hRanges`, `kRanges`, `lRanges`).
More concretely, calculate

``F_{hkl} = e^{- 2 \\pi i (x h + y k + z l)}}``

`x`, `y`, and `z` do not have to lie on any grid and are assumed to be `Vector{Real}`.
'hRanges', 'kRanges' and 'lRanges' are not individual points, but are `Vector{StepRangeLen}`,
that together, define the grid to sample reciprocal space over. In general, this will be
faster than a full discrete Fourier transform (with ``O(n^2)`` operations) because it uses
an NUFFT.
"""
function atomSimulateElectricField(x, y, z, hRanges, kRanges, lRanges)
    x = CuArray{Float64}(x)
    y = CuArray{Float64}(y)
    z = CuArray{Float64}(z)

    elecFields = Array{ComplexF64, 3}[]
    recSupport = Array{Float64, 3}[]
    Gs = Vector{Float64}[]
    boxSize = 1.0/Float64(hRanges[1].step)
    for i in 1:length(hRanges)
        Gh = Float64((hRanges[i].ref + hRanges[i].step * div(hRanges[i].len, 2)) / hRanges[i].step)
        Gk = Float64((kRanges[i].ref + kRanges[i].step * div(kRanges[i].len, 2)) / kRanges[i].step)
        Gl = Float64((lRanges[i].ref + lRanges[i].step * div(lRanges[i].len, 2)) / lRanges[i].step)
        h = zeros(0,0,0)
        k = zeros(0,0,0)
        l = zeros(0,0,0)
        G = [Gh, Gk, Gl]
        state = BcdiCore.AtomicState(
            "L2", false, 
            zeros(hRanges[i].len,kRanges[i].len,lRanges[i].len), 
            G, h, k, l, trues(0,0,0)
        )
        BcdiCore.setpts!(
            state,
            x .* Float64(hRanges[i].step) .* 2 .* pi, 
            y .* Float64(kRanges[i].step) .* 2 .* pi, 
            z .* Float64(lRanges[i].step) .* 2 .* pi,
            false
        )
        BcdiCore.forwardProp(state, true)
        push!(elecFields, Array(state.recipSpace))
        push!(recSupport, ones(Bool, size(elecFields[end])))
        push!(Gs, G)
    end
    return elecFields, recSupport, Gs, boxSize
end

"""
    atomSimulateDiffraction(x, y, z, hRanges, kRanges, lRanges)

Simulate diffraction patterns for a group of atoms (`x`, `y`, and `z`) on
a sequence of grids in reciprocal space (`hRanges`, `kRanges`, `lRanges`).
More concretely, obtain samples from a Poisson distribution that satisfy

``I_hkl} \\overset{ind}{\\sim} Pois(F_{hkl})``

where 

``F_{hkl} = e^{- 2 \\pi i (x h + y k + z l)}}``

`x`, `y`, and `z` do not have to lie on any grid and are assumed to be `Vector{Real}`.
'hRanges', 'kRanges' and 'lRanges' are not individual points, but are `Vector{StepRangeLen}`,
that together, define the grid to sample reciprocal space over. In general, this will be 
faster than a full discrete Fourier transform (with ``O(n^2)`` operations) because it uses
an NUFFT.
"""
function atomSimulateDiffraction(x, y, z, hRanges, kRanges, lRanges, numPhotons; seed=nothing)
    elecFields, recSupport, Gs, boxSize = simulateElectricField(x, y, z, hRanges, kRanges, lRanges)
    intens = Array{Int64, 3}[]
    if seed != nothing
        Random.seed!(seed)
    end
    for i in 1:length(elecFields)
        c = numPhotons[i] / mapreduce(abs2, +, elecFields[i])
        push!(intens, rand.(Poisson.(Array(c .* abs2.(elecFields[i])))))
    end
    return intens, recSupport, Gs, boxSize
end

"""
    relaxCrystal(x, y, z, lmpOptions, potentialName)

Use LAMMPS to relax the supplied atom positions (`x`, `y`, and `z`). `lmpOptions`
defines command line options to pass to `LAMMPS` and the `potentialName` defines
the interatomic potential used in the `LAMMPS` relaxation.
"""
function relaxCrystal(x, y, z, lmpOptions, potentialName)
    lo = min(minimum(x), minimum(y), minimum(z)) - 1
    hi = max(maximum(x), maximum(y), maximum(z)) + 1
    commandsInit = [
        "units metal",
        "dimension 3",
        "boundary f f f",
        "atom_style atomic",
        "atom_modify map array",
        "region box block $(lo) $(hi) $(lo) $(hi) $(lo) $(hi)",
        "create_box 1 box"
    ]
    commandsRun = [
        "change_box all boundary s s s",
        "pair_style eam/alloy",
        "pair_coeff * * $(potentialName)",
        "neighbor 2.0 bin",
        "neigh_modify every 1 delay 0 check yes",
        "min_style cg",
        "minimize 1e-25 1e-25 5000 10000",
    ]

    numAtoms = length(x)
    
    lmp = LMP(lmpOptions)
    indices = Int32.(collect(1:numAtoms))
    types  = ones(Int32, numAtoms)
    lammpsPositions = zeros(Float64, 3, numAtoms)
    lammpsPositions[1,:] .= x
    lammpsPositions[2,:] .= y
    lammpsPositions[3,:] .= z

    for com in commandsInit
        LAMMPS.command(lmp, com)
    end

    LAMMPS.create_atoms(lmp, lammpsPositions, indices, types, bexpand=true)

    for com in commandsRun
        LAMMPS.command(lmp, com)
    end
    
    indsPerm = invperm(Int64.(LAMMPS.extract_atom(lmp, "id", LAMMPS.LAMMPS_INT)))
    newPos = Float64.(LAMMPS.extract_atom(
        lmp, "x", LAMMPS.LAMMPS_DOUBLE_2D
    ))

    @views x .= newPos[1, indsPerm]
    @views y .= newPos[2, indsPerm]
    @views z .= newPos[3, indsPerm]
end
