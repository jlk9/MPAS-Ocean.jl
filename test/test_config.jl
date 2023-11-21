using Test
import YAML 

include("../src/infra/Config.jl")


# I'm not sure this is worth it. 
@test typeof(namelist_config(Dict{}())) == namelist_config
@test typeof(streams_config(Dict{}())) == streams_config 

dict_type = Dict{String, Any}

namelist_test = dict_type("hmix" => dict_type("config_hmix_ref_cell_width" => 30000.0, 
                                              "config_apvm_scale_factor" => 0.0, 
                                              "config_hmix_scaleWithMesh" => false, 
                                              "config_maxMeshDensity" => -1.0, 
                                              "config_hmix_use_ref_cell_width" => false),

                         "time_management" => dict_type("config_run_duration" => "0010_00:00:00", 
                                                        "config_do_restart" => false, 
                                                        "config_calendar_type" => "noleap", 
                                                        "config_restart_timestamp_name" => "Restart_timestamp", 
                                                        "config_stop_time" => "none", 
                                                        "config_output_reference_time" => "0001-01-01_00:00:00", 
                                                        "config_start_time" => "0001-01-01_00:00:00"))
# Test dictionary to write to temp file
streams_test = dict_type("streams" => dict_type(
                                      dict_type("mesh" => dict_type("type" => "input", 
                                                                    "filename_template"=>"init.nc", 
                                                                    "input_interval" => "initial_only"), 

                                                "output" => dict_type("filename_interval" => "01-00-00_00:00:00", 
                                                                      "clobber_mode" => "truncate", 
                                                                      "filename_template" => "output.nc", 
                                                                      "output_interval" => "0000_10:00:00", 
                                                                      "contents" => ["xtime", "normalVelocity", "layerThickness", "ssh"], 
                                                                      "precision" => "double", 
                                                                      "type" => "output", 
                                                                      "reference_time" => "0001-01-01_00:00:00"))))




# pack the namelist and stream dictionaries into to a test YAML dictionary 
YAML_test = dict_type("omega" => merge(namelist_test, streams_test))

mktemp("") do path, io
    # write the test dictionary to the temporary file 
    YAML.write_file(path, YAML_test)
    
    # read the test configuration file from the temporary file 
    config = ConfigRead(path)

    # try parsing a block 
    hmixConfig = ConfigGet(config.namelist, "hmix")
    # try getting an actual varibale value 
    hmixRefCellWidth = ConfigGet(hmixConfig, "config_hmix_ref_cell_width")

    @test hmixRefCellWidth ≈ 30000.0

end 



