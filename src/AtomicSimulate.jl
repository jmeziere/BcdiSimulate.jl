"""
    atomSimulateElectricField(x, y, z, hRanges, kRanges, lRanges)

Simulate the electric field for a group of atoms (`x`, `y`, and `z`) on
a sequence of grids in reciprocal space (`hRanges`, `kRanges`, `lRanges`).
More concretely, calculate

```math
F_{hkl} = \\sum_j e^{- 2 \\pi i (x_j h + y_j k + z_j l)}
```

`x`, `y`, and `z` do not have to lie on any grid and are assumed to be `Vector{Real}`.
`hRanges`, `kRanges` and `lRanges` are not individual points, but are `Vector{StepRangeLen}`,
that together, define the grid to sample reciprocal space over. In general, this will be
faster than a full discrete Fourier transform (with ``O(n^2)`` operations) because it uses
an NUFFT.
"""
function atomSimulateElectricField(x, y, z, hRanges, kRanges, lRanges, rotations)
    x = CuArray{Float64}(x)
    y = CuArray{Float64}(y)
    z = CuArray{Float64}(z)

    elecFields = Array{ComplexF64, 3}[zeros(ComplexF64,hRanges[i].len,kRanges[i].len,lRanges[i].len) for i in 1:length(hRanges)]
    recSupport = Array{Float64, 3}[ones(Bool, size(elecFields[i])) for i in 1:length(hRanges)]
    GCens = Vector{Float64}[zeros(3) for i in 1:length(hRanges)]
    GMaxs = Vector{Float64}[zeros(3) for i in 1:length(hRanges)]
    boxSize = 1.0/Float64(hRanges[1].step)
    @sync for i in 1:length(hRanges)
        @async begin
            Gh = hRanges[i][div(hRanges[i].len, 2)+1] / Float64(hRanges[i].step)
            Gk = kRanges[i][div(kRanges[i].len, 2)+1] / Float64(kRanges[i].step)
            Gl = lRanges[i][div(lRanges[i].len, 2)+1] / Float64(lRanges[i].step)
            h = zeros(0,0,0)
            k = zeros(0,0,0)
            l = zeros(0,0,0)
            G = [Gh, Gk, Gl]
            state = BcdiCore.AtomicState(
                "L2", false, 
                zeros(hRanges[i].len,kRanges[i].len,lRanges[i].len), 
                G, h, k, l, trues(0,0,0)
            )
            rot = transpose(rotations[i])
            xp = rot[1,1] .* x .+ rot[1,2] .* y .+ rot[1,3] .* z
            yp = rot[2,1] .* x .+ rot[2,2] .* y .+ rot[2,3] .* z
            zp = rot[3,1] .* x .+ rot[3,2] .* y .+ rot[3,3] .* z
            BcdiCore.setpts!(
                state,
                xp .* Float64(hRanges[i].step) .* 2 .* pi, 
                yp .* Float64(kRanges[i].step) .* 2 .* pi, 
                zp .* Float64(lRanges[i].step) .* 2 .* pi,
                false
            )
            BcdiCore.forwardProp(state, true)
            copyto!(elecFields[i], state.recipSpace)
            GCens[i] .= rotations[i] * G
            GCens[i] .*= [hRanges[i].step,kRanges[i].step,lRanges[i].step]
            maxarg = argmax(abs2.(elecFields[i]))
            GMaxs[i] .= [hRanges[i][maxarg[1]], kRanges[i][maxarg[2]], lRanges[i][maxarg[3]]]
            GMaxs[i] .= rotations[i] * GMaxs[i]
            nothing  # JuliaLang/julia#40626
        end
    end
    return elecFields, recSupport, GCens, GMaxs, boxSize
end

"""
    atomSimulateDiffraction(x, y, z, hRanges, kRanges, lRanges, numPhotons; seed=nothing)

Simulate diffraction patterns for a group of atoms (`x`, `y`, and `z`) on
a sequence of grids in reciprocal space (`hRanges`, `kRanges`, `lRanges`).
More concretely, obtain samples from a Poisson distribution that satisfy

```math
I_{hkl} \\overset{ind}{\\sim} Pois(F_{hkl})
```

where 

```math
F_{hkl} = \\sum_j e^{- 2 \\pi i (x_j h + y_j k + z_j l)}
```

`x`, `y`, and `z` do not have to lie on any grid and are assumed to be `Vector{Real}`.
`hRanges`, `kRanges` and `lRanges` are not individual points, but are `Vector{StepRangeLen}`,
that together, define the grid to sample reciprocal space over. `numPhotons` defines the
number of photons that will, on average, be simulated, and `seed` is the rng seed. In general, 
this will be faster than a full discrete Fourier transform (with ``O(n^2)`` operations) 
because it uses an NUFFT.
"""
function atomSimulateDiffraction(x, y, z, hRanges, kRanges, lRanges, rotations, numPhotons; seed=nothing)
    elecFields, recSupport, GCens, GMaxs, boxSize = atomSimulateElectricField(x, y, z, hRanges, kRanges, lRanges, rotations)
    intens = Array{Int64, 3}[]
    if seed != nothing
        Random.seed!(seed)
    end
    for i in 1:length(elecFields)
        c = numPhotons[i] / mapreduce(abs2, +, elecFields[i])
        push!(intens, rand.(Poisson.(Array(c .* abs2.(elecFields[i])))))
        maxarg = argmax(intens[i])
        GMaxs[i] .= [hRanges[i][maxarg[1]], kRanges[i][maxarg[2]], lRanges[i][maxarg[3]]]
        GMaxs[i] .= rotations[i] * GMaxs[i]
    end
    return intens, recSupport, GCens, GMaxs, boxSize
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

function generateDisplacement(x, y, z, recipLatt, xRange, yRange, zRange)
    imDisp = CUDA.zeros(Float64, 3, xRange.len * yRange.len * zRange.len)

    x = CuArray{Float64}(x)
    y = CuArray{Float64}(y)
    z = CuArray{Float64}(z)
    recipLatt = CuArray{Float64}(recipLatt)

    atDisp = CUDA.zeros(Float64, 3, length(x))
    atDisp[1,:] .= x
    atDisp[2,:] .= y
    atDisp[3,:] .= z

    function getInd(x, y, z)
        return (
            floor(Int64, (x-Float64(xRange.ref)) / Float64(xRange.step)+1/2) + 1,
            floor(Int64, (y-Float64(yRange.ref)) / Float64(yRange.step)+1/2) + 1,
            floor(Int64, (z-Float64(zRange.ref)) / Float64(zRange.step)+1/2) + 1
        )
    end
    atInds = getInd.(x, y, z)
    imInds = cu(vec(CartesianIndex.(Iterators.product(1:xRange.len,1:yRange.len,1:zRange.len))))

    atDisp = recipLatt * atDisp
    atDisp .-= floor.(Int64, atDisp .- 0.5)
    atDisp = exp.(1im .* 2 .* pi .* atDisp)
        
    function aveDisp(imInd, atInds, atDisp)
        tmp = 0.0 + 0.0 * 1im
        count = 0
        for i in 1:length(atInds)
            if atInds[i] == imInd
                tmp += atDisp[i]
                count += 1
            end
        end
        if count != 0
            return angle(tmp)
        else
            return 0.0
        end
    end

    @views imDisp[1,:] .= aveDisp.(imInds, Ref(atInds), Ref(atDisp[1,:]))
    @views imDisp[2,:] .= aveDisp.(imInds, Ref(atInds), Ref(atDisp[2,:]))
    @views imDisp[3,:] .= aveDisp.(imInds, Ref(atInds), Ref(atDisp[3,:]))
    imDisp = recipLatt \ imDisp

    return 
        reshape(Array(imDisp[1,:]), xRange.len, yRange.len, zRange.len), 
        reshape(Array(imDisp[2,:]), xRange.len, yRange.len, zRange.len),
        reshape(Array(imDisp[3,:]), xRange.len, yRange.len, zRange.len)
end

function getRotations(from, to)
    rotations = []
    for i in 1:length(from)
        kp = cross(from[i],to[i])
        theta = acos(dot(from[i],to[i])/(norm(from[i])*norm(to[i])))
        k = kp/norm(kp)
        K = [
            0 -k[3] k[2];
            k[3] 0 -k[1];
            -k[2] k[1] 0;
        ]
        push!(rotations, I + sin(theta) * K + (1 - cos(theta)) * (K*K))
    end
    return rotations
end

function createGoldSample(box, numVoronoi)
    seed = create([4.078,4.078,4.078],"fcc","Au",[0,0],[[1,0,0],[0,1,0],[0,0,1]])
    nodes = [[box[1]/2,box[2]/2,box[3]/2,0,0,0]]
    for i in 1:numVoronoi-1
        push!(nodes, [
            rand() * box[1], rand() * box[2], rand() * box[3],
            rand() * 180 - 90, rand() * 180 - 90, rand() * 180 - 90
        ])
    end
    config = polycrystal(seed, box, nodes)
    select(config, "prop","grainID",1)
    select(config, "invert")
    removeatom(config, "select")
    center(config, "com")
    return config.P[1,:], config.P[2,:], config.P[3,:]
end

