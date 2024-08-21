using Test
using MOKA: ConfigRead, ConfigGet, GlobalConfig, yaml_config 

#=
omega:
  hmix:
    HmixString: Restart_timestamp
    HmixFloat: 1.234567890
    HmixNone: none
    HmixOn: true
    HmixOff: false
    HmixExp: 1.e25
    config_start_time: 0001-01-01_00:00:00
    config_run_duration: 0000_10:00:00
    config_output_reference_time: 0001-01-01_00:00:00
  hmix:
    config_test_logical: false
    config_test_float: -1.0
    config_test_exponent: 1.0e5
    config_test_integer: 10
    config_test_string: output_interval
  streams
    restart:
      type: input;output
      filename_template: restarts/restart.$Y-$M-$D_$h.$m.$s.nc
      filename_interval: output_interval
      reference_time: 0001-01-01_00:00:00
      clobber_mode: truncate
      input_interval: initial_only
      output_interval: 0005_00:00:00
    output:
      type: output
      filename_template: output.nc
      filename_interval: 01-00-00_00:00:00
      reference_time: 0001-01-01_00:00:00
      clobber_mode: truncate
      precision: double
      output_interval: 0000_10:00:00
      contents:
      - xtime
      - normalVelocity
      - layerThickness
      - ssh
=#

RefHmixString = "Restart_timestamp"
RefHmixFloat  = 1.234567890
RefHmixNone   = nothing
RefHmixOn     = true
RefHmixOff    = false
RefHmixExp    = 1.e25

path="test.yaml"

# Build up a reference configuration
ConfigMokaAll  = GlobalConfig()
ConfigMokaHmix = yaml_config()

#mktemp("") do path, io
    # read the test configuration file from the temporary file 
    config = ConfigRead(path)

    # try parsing a block 
    hmixConfig = ConfigGet(config.namelist, "hmix")
    # try getting an actual varibale value 
    hmixRefCellWidth = ConfigGet(hmixConfig, "config_hmix_ref_cell_width")

    @test hmixRefCellWidth ≈ 30000.0

#end 



