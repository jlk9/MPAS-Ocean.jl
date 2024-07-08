
import YAML: compose, Resolver, Composer, forward!, StreamStartEvent, compose_document, 
             StreamEndEvent, peek, DocumentStartEvent, TokenStream, Constructor, load, 
             EventStream, Node, construct_document
"""Types which inherit from `yaml_config` should have 
a `dict` attribute. 
"""
abstract type yaml_config end 

"""Making an alias for the type of dict attribute
to allow for more flexibility if it were to change 
in the future
"""
dict_type = Dict{Any, Any}


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
    config = open(filepath, "r") do input
        load_MPAS(input, MPAS_Custom_Constructor(); resolver=MPAS_Custom_Resolver())
    end 

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

# to do: Need to generalize this regex pattern to work for all the possible 
#        MPAS timestamps options. 
const timestamp_pat = r"^(?:
                    (?:(\d{1,4})-)?      (?# year)
                    (?:(\d\d?)-)?        (?# month)
                    (\d+)                (?# day)
                    )?
                    _?                  
                    (\d\d):              (?# hour)
                    (\d\d):              (?# minute)
                    (\d\d)               (?# second)
                    $"x 

function index_to_period(captures)
    # get the non-zero index 
    idx = findall(x->!iszero(x), captures)[1]

    # https://github.com/JuliaLang/julia/issues/18285#issuecomment-1153218675
    idx == 1 && return Year(captures[idx])
    idx == 2 && return Month(captures[idx])
    idx == 3 && return Day(captures[idx])
    
    idx == 4 && return Hour(captures[idx])
    idx == 5 && return Minute(captures[idx])
    idx == 6 && return Second(captures[idx])
end

function construct_MPAS_TimeStamp(constructor::YAML.Constructor, node::YAML.Node)
    value = YAML.construct_scalar(constructor, node)
    mat = match(timestamp_pat, value)
    if mat === nothing
        throw(YAML.ConstructorError(nothing, nothing,
            "could not make sense of timestamp format", node.start_mark))
    end
    
    # In the case where all groups are passed, should be a DateTime type 
    if all(mat.captures .!= nothing)
        h  = parse(Int, mat.captures[4])
        m  = parse(Int, mat.captures[5])
        s  = parse(Int, mat.captures[6])

        yr = parse(Int, mat.captures[1])
        mn = parse(Int, mat.captures[2])
        dy = parse(Int, mat.captures[3])
        
        if !any(iszero.((mn,dy)))
            return DateTime(yr, mn, dy, h, m, s)
        else 
            return "too complicated for right now"
        end
    end 
    
    # if element is equal to nothing, return Int(0). Otherwise 
    # parse the string as an Int
    # https://stackoverflow.com/a/54393947
    captures = [x==nothing ? 0::Int : parse(Int,x) for x in mat.captures]
    
    # in the case where everything is zero or nothing, except one field 
    # return a period corresponding to that field 
    if count(!iszero, captures) == 1 
        return index_to_period(captures)
    end 
    
    # Need more error handling for intermediate cases
    h  = parse(Int, mat.captures[4])
    m  = parse(Int, mat.captures[5])
    s  = parse(Int, mat.captures[6])

    # No Y/M/D info in timestamp
    if all(mat.captures[1:3] .=== nothing)
        return Time(h, m, s)
    elseif all(mat.captures[1:2] .=== nothing)
    # add case for time detlas in months
        
        if parse(Int, mat.captures[3]) == 0 
            # return period, NOT time 
            #return Hour(h) + Minute(m) + Second(s)
            return Time(h,m,s)
        else 
            return "time delta in days"
        end 
    end

    yr = parse(Int, mat.captures[1])
    mn = parse(Int, mat.captures[2])
    dy = parse(Int, mat.captures[3])


	# handle the interval options, year is allowed to be zero 	 
	# but month and day are not 
	if any(iszero.((mn,dy)))
		return "time delta in years"
	end 

    #return DateTime(yr, mn, dy, h, m, s)
end
 
function MPAS_Custom_Resolver()
    # Make is so that the resolver and the constructor regex pattern 
    # are both added globably. 
    MPAS_TimeStamp_Resolver = tuple("tag:MPAS_timestamp", timestamp_pat) 
 
    # Instansiate the Resolver struct, so that the defaults can be 
    # over written and passed to our custom `load` method 
    resolver = Resolver()

    # append the custom MPAS timestamp resolver to the end of the arrray
    # of regex patterns within the instance of the `Resolver`
    push!(resolver.implicit_resolvers, MPAS_TimeStamp_Resolver)

    resolver 
end 

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


function compose_MPAS(events; resolver::Union{Resolver, Nothing}=nothing)
    # if no resolver was passed, return default resolver. 
    # otherwise use resolver passed as kwarg 
    resolver = resolver ==  nothing ? Resolver : resolver 
    composer = Composer(events, Dict{String, Node}(), resolver)
    @assert typeof(forward!(composer.input)) == StreamStartEvent
    node = compose_document(composer)
    if typeof(peek(composer.input)) == StreamEndEvent
        forward!(composer.input)
    else 
        @assert typeof(peek(composer.input)) == DocumentStartEvent
    end 
    node
end 

load_MPAS(ts::TokenStream, constructor::Constructor; resolver::Resolver) = 
    construct_document(constructor, compose_MPAS(EventStream(ts); resolver=resolver))

load_MPAS(input::IO, constructor::Constructor; kwargs...) = load_MPAS(TokenStream(input), constructor; kwargs...)

