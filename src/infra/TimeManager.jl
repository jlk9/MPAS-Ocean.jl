# abstract alarm type 
abstract type AbstractAlarm end 

# creat a ESMF clock like structure 
mutable struct Clock 

    startTime::DateTime
    currTime::DateTime
    prevTime::Union{DateTime, Nothing}
    nextTime::DateTime
    timeStep::Period

    alarms::Dict{String, AbstractAlarm}

    # How does it know the stop time? Is that set as an alarm? 
    function Clock(startTime::DateTime, timeStep::Period)
        currTime = startTime 
        # at initialization there has been no prev. time 
        prevTime = nothing
        # could follow the CANGA approach; might be helpfull w/ restarts
        #prevTime = currTime - timeStep
        nextTime = currTime + timeStep
        # create an empty dictionary of alarms
        alarms = Dict{String, AbstractAlarm}()
         
        # return an instance of our structure 
        return new(startTime, currTime, prevTime, nextTime, timeStep, alarms)
    end 
end 

function setCurrentTime!(clock::Clock, inCurrTime::DateTime)
    # Check that new value doesn't precede start time 
    if inCurrTime < clock.startTime
        @error "Value of current time precedes start time" 
    else 
        clock.currTime = inCurrTime
        clock.prevTime = inCurrTime - clock.timeStep
        clock.nextTime = inCurrTime + clock.timeStep
    end
end

function changeTimeStep!(clock::Clock, timestep::Period)
    # Assign the new time step to this clock
    clock.timeStep = timestep
    # Update the next time based on new time step
    clock.nextTime = clock.currTime + timestep 
end 

function attachAlarm!(clock::Clock, alarm::AbstractAlarm)
    # use the alarms name to insert it in the dict of alarms
    clock.alarms[alarm.name] = alarm
end

function advance!(clock::Clock)
    # Advance clock attributes by one timestep. 
    clock.prevTime = clock.currTime
    clock.currTime = clock.nextTime 
    clock.nextTime = clock.currTime + clock.timeStep
    # Update status of any attached alarms via broadcasting
    # across the values of the alarms dictionary
    updateStatus!.(values(clock.alarms), clock.currTime)
end 

Base.show(io::IO, clock::Clock) = 
    print(io, "Simulation Clock with $(length(clock.alarms)) Alarms attached\n",
          "├── Start Time   : $(clock.startTime)\n",
          "├── Current Time : $(clock.currTime)\n",
          "├── Previous Time: $(clock.prevTime)\n",
          "├── Next Time    : $(clock.nextTime)\n",
          "└── Timestep     : $(clock.timeStep)")


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

# 
Alarm(name::String, alarmTime::DateTime) = OneTimeAlarm(name, alarmTime)
Alarm(name::String, alarmInterval::Period, intervalStart::DateTime) = 
    PeriodicAlarm(name, alarmInterval, intervalStart)

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
function reset!(alarm::OneTimeAlarm)
    stop!(alarm)
    alarm.stopped = true 
end

function reset!(alarm::OneTimeAlarm, inTime::DateTime)
    stop!(alarm)
    alarm.ringTime = inTime 
end

function reset!(alarm::PeriodicAlarm)
    stop!(alarm)
    # store the ringtime just turned off as the pervious ringtime
    alarm.ringTimePrev = alarm.ringTime
    # set the next ringTime
    alarm.ringTime = alarm.ringTimePrev + alarm.ringInterval
end 

function reset!(alarm::PeriodicAlarm, inTime::DateTime)
    stop!(alarm)

    # check that the input time is valid 
    if inTime < alarm.ringTime
        @error "input time less than the current ring time"
    else
        # increment until next ringtime is greater than input time
        while alarm.ringTime <= inTime 
            alarm.ringTimePrev = alarm.ringTime
            alarm.ringTime = alarm.ringTimePrev + alarm.ringInterval
        end
    end
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
