module sugarscape

using Agents, GLMakie, Random

@agent SugarSeeker GridAgent{2} begin
    vision::Int
    metabolic_rate::Int
    age::Int
    max_age::Int
    wealth::Int
end

function distances(position, sugar_peaks)
    all_distances = zeros(Int, length(sugar_peaks))
    for (index, peak) in enumerate(sugar_peaks)
        distance = round(Int, sqrt(sum((position .- peak) .^ 2)))
        all_distances[index] = distance
    end
    return minimum(all_distances)
end

function sugar_caps(dims, sugar_peaks, max_sugar, diameter=4)
    sugar_capacities = zeros(Int, dims)
    for i in 1:dims[1], j in 1:dims[2]
        sugar_capacities[i, j] = distances((i, j), sugar_peaks)
    end
    for i in 1:dims[1]
        for j in 1:dims[2]
            sugar_capacities[i, j] = max(0, max_sugar - sugar_capacities[i, j] ÷ diameter)
        end
    end
    return sugar_capacities
end

function sugarscape(;
    dims=(50, 50),
    sugar_peaks=((10, 40), (40, 10)),
    growth_rate=1,
    N=250,
    w0_dist=(5, 25),
    metabolic_rate_dist=(1, 4),
    vision_dist=(1, 6),
    max_age_dist=(60, 100),
    max_sugar=4,
    seed=42
)
    diameter = 6
    sugar_capacities = sugar_caps(dims, sugar_peaks, max_sugar, diameter)
    sugar_values = deepcopy(sugar_capacities)
    space = GridSpaceSingle(dims)
    properties = Dict(
        :growth_rate => growth_rate,
        :N => N,
        w0_dist => w0_dist,
        :metabolic_rate_dist => metabolic_rate_dist,
        :vision_dist => vision_dist,
        :max_age_dist => max_age_dist,
        :sugar_values => sugar_values,
        :sugar_capacities => sugar_capacities,
    )
    model = AgentBasedModel(
        SugarSeeker2,
        space,
        scheduler=Schedulers.Randomly,
        properties=properties,
        rng=MersenneTwister(seed)
    )
    for _ in 1:N
        add_agent_single!(
            model,
            rand(model.rng, vision_dist[1]:vision_dist[2]),
            rand(model.rng, metabolic_rate_dist[1]:metabolic_rate_dist[2]),
            0,
            rand(model.rng, max_age_dist[1]:max_age_dist[2]),
            rand(model.rng, w0_dist[1]:w0_dist[2]),
        )
    end
    return model
end

function model_step!(model)
    # At each position, sugar grows back at a rate of α units per time-step, up to cell's capacity c
    @inbounds for pos in positions(model)
        if model.sugar_values[pos...] < model.sugar_capacities[pos...]
            model.sugar_values[pos...] += model.growth_rate
        end
    end
end

function agent_step!(agent, model)
    move_and_collect!(agent, model)
    replacement!(agent, model)
end

function move_and_collect!(agent, model)
    # Go through all unoccupied positions within vision, and consider the empty ones.
    # From those, identify the one with greatest amount of sugar, and go there.
    max_sugar_pos = agent.pos
    max_sugar = model.sugar_values[max_sugar_pos...]
    for pos in nearby_positions(agent, model, agent.vision)
        isempty(pos, model) || continue
        sugar = model.sugar_values[pos...]
        if sugar > max_sugar
            max_sugar = sugar
            max_sugar_pos = pos
        end
    end
    # Move to the max sugar position (which could be where we are already)
    move_agent!(agent, max_sugar_pos, model)
    # Collect the sugar there and update wealth (collected - consumed)
    agent.wealth += (model.sugar_values[max_sugar_pos...] - agent.metabolic_rate)
    model.sugar_values[max_sugar_pos...] = 0

    agent.age += 1
    return
end

function replacement!(agent, model)
    if agent.wealth ≤ 0 || agent.age ≥ agent.max_age
        remove_agent!(agent, model)
        # Whenever an agent dies, a young one is added to a random empty position.
        add_agent_single!(
            model,
            rand(model.rng, model.vision_dist[1]:model.vision_dist[2]),
            rand(model.rng, model.metabolic_rate_dist[1]:model.metabolic_rate_dist[2]),
            0,
            rand(model.rng, model.max_age_dist[1]:model.max_age_dist[2]),
            rand(model.rng, model.w0_dist[1]:model.w0_dist[2]),
        )
    end
end

model = sugarscape()

fig, ax, abmobs = abmplot(
    model;
    agent_step!,
    model_step!
)

fig

end # module sugarscape
