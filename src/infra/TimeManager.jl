# abstract alarm type 
abstract type abstract_alarm end

# creat a ESMF clock like structure 
mutable struct Clock 

    startTime::DateTime
    currTime::DateTime
    prevTime::Union{DateTime, Nothing}
    nextTime::DateTime
    timeStep::Period

    # How does it know the stop time? Is that set as an alarm? 
    function Clock(startTime::DateTime, timeStep::Period)
        currTime = startTime 
        prevTime = nothing # at the start of the sim. there has been no prev. time 
        nextTime = currTime + timeStep
        
        # return an instance of our structure 
        return new(startTime, currTime, prevTime, nextTime, timeStep)
    end 
end 

function advance!(clock::Clock)
    clock.prevTime = clock.currTime
    clock.currTime = clock.nextTime 
    clock.nextTime = clock.currTime + clock.timeStep
end 


function mpas_create_clock(timeStep, startTime; stopTime=nothing, runDuration=nothing)
    
    if !isnothing(runDuration)
        stop_time = startTime + runDuration
        if !isnothing(stopTime)
            stopTime != stop_time || throw("stopTime and runDuration are inconsistent") 
        end 
    elseif !isnothing(stopTime)
        stop_time = stopTime
    else
        throw(" neither stopTime nor runDuration are specified")
    end 
        
    clock = Clock(startTime, timeStep)

    return clock 
end 
