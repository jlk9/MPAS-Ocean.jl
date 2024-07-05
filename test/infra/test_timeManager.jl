using Test
using Dates
using UnPack
using MOKA: Clock, changeTimeStep!, attachAlarm!, advance!,
            OneTimeAlarm, PeriodicAlarm, isRinging, updateStatus!,
            rename!, stop!, reset!, mpas_create_clock, Alarm, setCurrentTime! 
# alernatively should make a module for timeMagament / infra 
# to deal withe the namespace managment there

# Create an initial model clock with a start time of 2000-01-01_00:00:00 and
# timestep of 1 hour
time0 = DateTime(2000, 1, 1, 0, 0, 0)
timestep = Hour(1)
modelClock = Clock(time0, timestep)

# Test some basic retrival functions 
@test time0 == modelClock.currTime
@test timestep == modelClock.timeStep

# Define ringtime for one-time alarms
time2020Mar01 = DateTime(2020, 3, 01, 0, 0, 0)
time2019Aug24 = DateTime(2019, 8, 24, 0, 0, 0)
timeNewYear2020 = DateTime(2020, 1, 01, 0, 0, 0)

# Define the one-time alarms
alarm2020Mar01 = Alarm("2020-03-01", time2020Mar01) 
alarm2019Aug24 = Alarm("2019-08-24", time2019Aug24) 
alarmNewYear2020 = Alarm("New Year 2020", timeNewYear2020) 

# Define intervals for periodic alarms 
interval20Min    = Minute(20)
interval1Hour    = Hour(1)
interval6Hour    = Hour(6) 
intervalDaily    = Day(1)
intervalMonthly  = Month(1)
intervalAnnually = Year(1)

# Define the periodic alarms
alarmEvery20Min  = Alarm("Every 20 minutes", interval20Min   ,time0)
alarmEvery1Hours = Alarm("Every hour",       interval1Hour   ,time0)
alarmEvery6Hours = Alarm("Every 6 hours",    interval6Hour   ,time0)
alarmEveryDay    = Alarm("Every day",        intervalDaily   ,time0)
alarmEveryMonth  = Alarm("Every month",      intervalMonthly ,time0)
alarmEveryYear   = Alarm("Every year",       intervalAnnually,time0)

# Attach the one-time alarms to the clock
attachAlarm!(modelClock, alarm2020Mar01)
attachAlarm!(modelClock, alarm2019Aug24)
attachAlarm!(modelClock, alarmNewYear2020)

# Attach the periodic alarms to the clock
attachAlarm!(modelClock, alarmEvery20Min)
attachAlarm!(modelClock, alarmEvery1Hours)
attachAlarm!(modelClock, alarmEvery6Hours)
attachAlarm!(modelClock, alarmEveryDay)
attachAlarm!(modelClock, alarmEveryMonth)
attachAlarm!(modelClock, alarmEveryYear)

# Test changing the timestep
changeTimeStep!(modelClock, interval20Min)
# Retrive the timestep from the clock
stepCheck = modelClock.timeStep

# Check that the timestep was updated properly
@test stepCheck == interval20Min

# Test chaning the current time

# Let's come up with a new currTime
testCurrTime = DateTime(2019, 01, 01, 00, 00, 00)
# Hardcode the prev/next times based on the 20 minutes timestep (set above)
testPrevTime = DateTime(2018, 12, 31, 23, 40, 00) 
testNextTime = DateTime(2019, 01, 01, 00, 20, 00)

setCurrentTime!(modelClock, testCurrTime)

@test modelClock.currTime == testCurrTime
@test modelClock.prevTime == testPrevTime
@test modelClock.nextTime == testNextTime

# Update the periodic alarms to the new current time 
reset!(alarmEvery20Min,  testCurrTime)
reset!(alarmEvery1Hours, testCurrTime)
reset!(alarmEvery6Hours, testCurrTime)
reset!(alarmEveryDay,    testCurrTime)
reset!(alarmEveryMonth,  testCurrTime)
reset!(alarmEveryYear,   testCurrTime)

# simulation the forward integration of model by advancing the clock forward in
# time and checking alarms. Integrate forward for 2 years checking alarms every
# 20 minutes.
stopTime = DateTime(2021, 01, 01, 00, 00, 00)

while modelClock.currTime <= stopTime
    # advance one timestep
    advance!(modelClock)

    # get the current model time 
    currTime = modelClock.currTime
    
    # // Check one time Alarms
    if currTime == time2020Mar01
        @test isRinging(alarm2020Mar01)
        stop!(alarm2020Mar01)
    end

    if currTime == time2019Aug24
        @test isRinging(alarm2019Aug24)
        stop!(alarm2019Aug24)
    end

    if currTime == timeNewYear2020
        @test isRinging(alarmNewYear2020)
        stop!(alarmNewYear2020)
    end
    # //
    
    # get the various period component that make up the current DateTime
    months  = Month(currTime).value
    days    = Day(currTime).value
    hours   = Hour(currTime).value
    minutes = Minute(currTime).value
    seconds = Second(currTime).value

    # // Check 20 minute alarm
    ringCheck = isRinging(alarmEvery20Min)
    
    # Only check this test for the 23rd hour of each day, otherwise we check
    # this condition VERY frequently for a 2 year simulation
    if minutes % 20 == 0 && seconds == 0
       @test isRinging(alarmEvery20Min)
    end
     
    # // Check hourly alarm
    if minutes == 0 && seconds == 0
        @test isRinging(alarmEvery1Hours)
    end
    # //  

    # // Check the 6 hour alarm
    if hours % 6 == 0 && minutes == 0 && seconds == 0
        @test isRinging(alarmEvery6Hours)
    end
    # //
    
    # // Check the daily alarm 
    if  hours == 0 && minutes == 0 && seconds == 0
        @test isRinging(alarmEveryDay)
    end 
    # //
   
    # // Check the monthly alarm
    if days == 1 && hour == 0 && minutes == 0 && seconds == 0
        @test isRinging(alarmEveryMonth)
    end
    # //
    
    # // Check the yearly alarm
    if months == 1 && days == 1 && hours == 0 && minutes == 0 && seconds == 0
        @test isRinging(alarmEveryYear)
    end
    # //
end
