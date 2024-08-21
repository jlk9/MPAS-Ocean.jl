using Test
using Dates
using MOKA: ConfigRead, ConfigGet, GlobalConfig, yaml_config,
            ConfigSet, ConfigAdd

ref_hmix_String = "Restart_timestamp"
ref_hmix_Float  = 1.234567890
ref_hmix_None   = "none"
ref_hmix_On     = true
ref_hmix_Off    = false
ref_hmix_Exp    = 1.e25

# read the test configuration file from the temporary file 
config = ConfigRead("test.yaml")

# parse 
hmixConfig      = ConfigGet(config.namelist, "hmix")
intervalsConfig = ConfigGet(config.streams,  "intervals")
datetimesConfig = ConfigGet(config.streams,  "datetimes")

# Test parsing various datatypes
@test ref_hmix_String == ConfigGet(hmixConfig, "hmix_String") 
@test ref_hmix_Float  == ConfigGet(hmixConfig, "hmix_Float") 
@test ref_hmix_None   == ConfigGet(hmixConfig, "hmix_None") 
@test ref_hmix_On     == ConfigGet(hmixConfig, "hmix_On") 
@test ref_hmix_Off    == ConfigGet(hmixConfig, "hmix_Off") 
@test ref_hmix_Exp    == ConfigGet(hmixConfig, "hmix_Exp") 

# Test parsing Periods
@test Year(1)   == ConfigGet(intervalsConfig, "yearly_interval")
@test Month(2)  == ConfigGet(intervalsConfig, "monthly_interval")
@test Day(3)    == ConfigGet(intervalsConfig, "daily_interval")
@test Hour(4)   == ConfigGet(intervalsConfig, "hourly_interval")
@test Minute(5) == ConfigGet(intervalsConfig, "minutes_interval")
@test Second(6) == ConfigGet(intervalsConfig, "seconds_interval")

# Test parsing DateTimes
@test DateTime(1,1,1,0,0,0) == ConfigGet(datetimesConfig, "NO_HMS")
@test DateTime(1,1,1,2,0,0) == ConfigGet(datetimesConfig, "NO_MS")  
@test DateTime(1,1,1,2,3,0) == ConfigGet(datetimesConfig, "NO_S")
@test DateTime(1,1,1,0,3,4) == ConfigGet(datetimesConfig, "NO_H")   
@test DateTime(1,1,1,0,0,4) == ConfigGet(datetimesConfig, "NO_HM")  
@test DateTime(1,1,1,0,3,0) == ConfigGet(datetimesConfig, "NO_HS")  
@test DateTime(1,1,1,2,3,4) == ConfigGet(datetimesConfig, "ALL_HMS")
