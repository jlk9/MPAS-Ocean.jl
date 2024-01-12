# abstract alarm type 
abstract type AbstractAlarm end 

# creat a ESMF clock like structure 
mutable struct Clock 

    startTime::DateTime
    currTime::DateTime
    prevTime::Union{DateTime, Nothing}
    nextTime::DateTime
    timeStep::Period

    alarms::Vector{AbstractAlarm}

    # How does it know the stop time? Is that set as an alarm? 
    function Clock(startTime::DateTime, timeStep::Period)
        currTime = startTime 
        prevTime = nothing # at the start of the sim. there has been no prev. time 
        nextTime = currTime + timeStep
        
        # return an instance of our structure 
        return new(startTime, currTime, prevTime, nextTime, timeStep, AbstractAlarm[])
    end 
end 

function attachAlarm!(clock::Clock, alarm::AbstractAlarm)
    push!(clock.alarms, alarm)
end

function advance!(clock::Clock)
    # Advance clock attributes by one timestep. 
    clock.prevTime = clock.currTime
    clock.currTime = clock.nextTime 
    clock.nextTime = clock.currTime + clock.timeStep

    # Update status of any attached alarms (via broadcasting)
    updateStatus!.(clock.alarms, clock.currTime)
end 


# create new abstract struct similar to alarms, which contains infromation
# about what to do when the alarm is ringing. Then a single function 
# e.g. `stopringing` can be call for each alarm, which does the various 
# things needed to be done (e.g. io, forcing, restart, analysis) when an alarm rings. 
# Find out in the current code what alarms are used for to think about the 
# type interface

mutable struct OneTimeAlarm{S,B,DT} <: AbstractAlarm
    name::S       # name of the alarm 

    ringing::B      # alarm is currently ringing
    stopped::B      # alarm had been stopped and not reset

    ringTime::DT # time at/after which alarm rings 

    function OneTimeAlarm(name::String, alarmTime::DateTime)
        return new{String, Bool, DateTime}(name, false, false, alarmTime)
    end 
end 


mutable struct PeriodicAlarm{S,B,DT,P} <: AbstractAlarm 
    name::S                           # name of the alarm 

    ringing::B                         # alarm is currently ringing
    stopped::B                          # alarm had been stopped and not reset

    ringTime::DT                     # time at/after which alarm rings 
    ringInterval::P                   # interval at which this alarm rings
    ringTimePrev::Union{Nothing, DT} # previous alaram time for periodic alarms
    
    # if Period is closed inteval  (c.f.open), then how do you deal with the first timestep since the 
    # `updateStatus` method is called at the end of the timestep, so if the alarm fall on the 
    # first timestep it is always skipped. 
    function PeriodicAlarm(name::String, alarmInterval::Period, intervalStart::DateTime)
        # NOTE: should alarm ring on interval start? or only after the first interval 
        #       could be optional keyword in construtor based on option from the `io` 
        #       section of the namelist 
        ringTime = intervalStart + alarmInterval 
        
        return new{String, Bool, DateTime, Period}(name, false, false, ringTime, alarmInterval, nothing)
    end
end


# Methods that work for all types of alarms 
function isRinging(alarm::AbstractAlarm)
    return alarm.ringing
end

function updateStatus!(alarm::AbstractAlarm, currentTime::DateTime)
    alarm.ringTime == currentTime && (alarm.ringing = true)
end 

function rename!(alarm::AbstractAlarm, newName::String)
    alarm.name = newName
end

function stop!(alarm::AbstractAlarm)
    alarm.ringing = false 
end


# Methods that have type specific behavior 
function reset!(alarm::OneTimeAlarm; inTime::Union{Nothing, DateTime}=nothing)
    stop!(alarm)

    if !isnothing(inTime)
        alarm.ringTime = inTime
    else 
        alarm.stopped = true 
    end 
end

# NOTE: Should reset method for PeriodicAlarms accepct a new ringTime?? 
function reset!(alarm::PeriodicAlarm)
    stop!(alarm)

    # store the ringtime just turned off as the pervious ringtime
    alarm.ringTimePrev = alarm.ringTime
    # set the next ringTime
    alarm.ringTime = alarm.ringTimePrev + alarm.ringInterval
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
