# Boggart

### To run online tests:
First run test_on_internet/init_all_servers.py to enable video download and logging from localhost. Make sure logging directoy exists inside 'servers' directory (configure name of logging directory in 'logging_server.py').
To run all the tested algorithms (Boggart, Pensieve, MPC, RateBased, BufferBased, Bola) concurrently, run test_on_internet/run_all_in_internet.py.
To test one algorithm run test_on_internet/communication.py [ALGORITHM].

### To run tests on simulator:
Inside run_all.py, configure list of desired ABR algorithms, QoE function, traces directory, and logging directory.
Then run run_all.py.

A single trace file should be a .json file, with an array which consists from dictionaries of the form:
{"duration_ms": x, "bandwidth_kbps": y, "latency_ms": z} ---> bandwidth of y kbps was experienced during x seconds of playback, with latency of z between server and client. For more information see https://github.com/UMass-LIDS/sabre/blob/master/src/sabre.py.
A trace example is provided in 'one_trace' folder.

### To train new Boggart model:
in session.py set abr_type = BOGGART, anv_type = TRAIN, and trace_dir, reward_type, LOG_DIR, save_boggart_model to desired values.
run session.py


The envornment simulator and pensieve model files are as in https://github.com/hongzimao/pensieve. 
