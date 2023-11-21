using YAML

"""Types which inherit from `yaml_config` should have 
a `dict` attribute. 
"""
abstract type yaml_config end 

"""Making an alias for the type of dict attribute
to allow for more flexibility if it were to change 
in the future
"""
dict_type = Dict{String, Any}


"""
Following Omega specs, define method to get

Allow the type info to be passed so that it 
can be used to create a new instance if string 
corresponds to header, not a config option. 
"""
function ConfigGet(d::C, s::String) where {C<:yaml_config}
    # could call exists function here
    
    # access the underlying dictionary contained within the object
    # and use key (string) to get config info 
    c = d.dict[s]
    
    # if a dictionary is returned that means we have reached the bottom 
    # level of the yaml tree, instead return a new instance of the 
    # configuration struct. 
    if typeof(c) == dict_type
        return C(c) 
    else
        return c
    end
end


#= DEFINE ALL THE SHARED METHODS HERE: 
    1. config get 
    2. config set (will require mutable struct) 
    3. config add (will require mutable struct)  
    4. config exists
    5. config write 
=# 



"""
As these types inherit from `yaml_config` they need
a `dict` attribute 
"""
struct namelist_config <: yaml_config
    dict::dict_type
end 

struct streams_config <: yaml_config
    dict::dict_type
end 


"""
Struct which stores the namelist and streams config strucutres
"""
struct GlobalConfig
    namelist::namelist_config
    streams::streams_config
end 


function ConfigRead(filepath::AbstractString)
    #=
    this function SHOULD: 
        1. check that the input filepath exists
        2. return a instance of the Config datastructure (which I guess has to be mutable)
    =#

    # check that the config YAML file exists
    if !isfile(filepath)
        # TO DO: Use logging library to write error message 
        error("YAML configuration file does not exist")
    end

    # load YAML file where dictionary keys are forced to types defined in 
    # the dictionary above. 
    config = YAML.load_file(filepath, dicttype=dict_type)
    
    # Extract the "streams" dictionary from the namelist dictionary 
    streams = pop!(config["omega"], "streams")
    # Extract the "namelist" dictionary from the global YAML file 
    namelist = pop!(config, "omega")

    # Create the instances of the namelist and streams configuration structs
    name = namelist_config(namelist)
    stream = streams_config(streams)

    # Pack the namelist and streams config. structs into the global config struct 
    return GlobalConfig(name, stream)
end


# Make is so that the resolver and the constructor regex pattern 
# are both added globably. 
MPAS_TimeStamp_Resolver = tuple("tag:MPAS_timestamp", timestamp_pat) 
# append the custom MPAS timestamp resolver to the end of the arrray 
append!(YAML.default_implicit_resolvers, [MPAS_TimeStamp_Resolver])

"""Create a custom constructor object, which is able to parse 
the MPAS timestamp format. 
"""
function MPAS_Custom_Constructor()
    # Get the deafault constructors
	yaml_constructors = copy(YAML.default_yaml_constructors)
	# Add a new constructor, for the MPAS timestamps  
	yaml_constructors["tag:MPAS_timestamp"] = construct_MPAS_TimeStamp
	# return an Construcutor object, which will parse the MPAS timestamps
	YAML.Constructor(yaml_constructors)
end 


# TO DO: Need to generalize this regex pattern to work for all the possible 
#        MPAS timestamps options. 
timestamp_pat =
    r"^(\d{4})-    (?# year)
       (\d\d?)-    (?# month)
       (\d\d?)     (?# day)
      (?:
		(?:_?)
        (\d\d?):      (?# hour)
        (\d\d):       (?# minute)
        (\d\d)        (?# second)
        (?:\.(\d*))?  (?# fraction)
        (?:
          [ \t]*(Z|(?:[+\-])(\d\d?)
            (?:
                :(\d\d)
            )?)
        )?
      )?$"x

function construct_MPAS_TimeStamp(constructor::YAML.Constructor, node::YAML.Node)
    value = YAML.construct_scalar(constructor, node)
    mat = match(timestamp_pat, value)
    if mat === nothing
        throw(YAML.ConstructorError(nothing, nothing,
            "could not make sense of timestamp format", node.start_mark))
    end

    yr = parse(Int, mat.captures[1])
    mn = parse(Int, mat.captures[2])
    dy = parse(Int, mat.captures[3])

	# handle the interval options, year is allowed to be zero 	 
	# but month and day are not 
	if any(iszero.((mn,dy)))
		return "CANT PARSE INTERVALS YET"
	end 

    if mat.captures[4] === nothing
        return Date(yr, mn, dy)
    end

    h = parse(Int, mat.captures[4])
    m = parse(Int, mat.captures[5])
    s = parse(Int, mat.captures[6])
	
	
    if mat.captures[7] === nothing
        return DateTime(yr, mn, dy, h, m, s)
    end

    ms = 0
    if mat.captures[7] !== nothing
        ms = mat.captures[7]
        if length(ms) > 3
            ms = ms[1:3]
        end
        ms = parse(Int, string(ms, repeat("0", 3 - length(ms))))
    end

    delta_hr = 0
    delta_mn = 0

    if mat.captures[9] !== nothing
        delta_hr = parse(Int, mat.captures[9])
    end

    if mat.captures[10] !== nothing
        delta_mn = parse(Int, mat.captures[10])
    end

    # TODO: Also, I'm not sure if there is a way to numerically set the timezone
    # in Calendar.

    return DateTime(yr, mn, dy, h, m, s, ms)
end

