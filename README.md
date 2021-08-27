# rapid-ip-checker (ric)

ric (Rapid IP Checker) is a tool that can be used to check whether a list of input Host/Network IP addresses (IPv4 and IPv6), is part of a large list of network ranges, using the parallel computing capabilities of general purpose graphics processing units (GP-GPUs).

ric leverages the following for this :
* *Python 3*
* *Compute Unified Device Architecture (CUDA) :*<br/>Nvidia's parallel computing platform and application programming interface, which is an extension of C, with special constructs to support parallel programming.
* *Numba :*<br/>A JIT compiler for Python, that supports CUDA by directly compiling a restricted subset of Python code into CUDA kernels.
* *RAPIDS APIS and Libraries (https://rapids.ai/) :* <br/>An opensource GPU data science framework.

### The Cuda Kernel :
The task that is compiled as GPU code, and run in parallel across the GPU cores (the Cuda Kernel), is the checking of whether a single IP address is within a network CIDR.

IP addresses can be represented as an integer using the dotted decimal format :

| Dotted Decimal Format | Integer Equivalent |
|:-:	|:-:	|
| 0.0.0.0 	| 0 |
| 0.0.0.1 	| 1 |
| 0.0.0.2 	| 2 |
| ... 	| ... |
| 255.255.255.255 | 4294967295 |

The mask can be indicate the number of hosts in a network :

```(2**(32-mask))-1```

Now, checking if an IP is in a network, becomes an integer comparison problem. This comparison is implemented as a Cuda Kernel, using Numba, that is run in parallel on the GPU cores.

## Setup :
### Create a conda environment "ric-rapids-0.19" :
`conda create -n ric-rapids-0.19 -c rapidsai -c nvidia -c conda-forge -c defaults cudf=0.19 python=3.7 cudatoolkit=11.0 numba numpy`

### Activate the environment :
`conda activate ric-rapids-0.19`

### Create a bash alias for the program :
```
alias ric="python ./ric.py"; source ~/.bashrc ; alias > ~/.bash_aliases`
alias ric6="python ./ric6.py"; source ~/.bashrc ; alias > ~/.bash_aliases`
```

### Run the Program :
For IPv4 addresses : `ric -i input_ipv4_file.txt -t target_ipv4_file.txt`<br />
For IPv6 addresses : `ric6 -i input_ipv6_file.txt -t target_ipv6_file.txt`

### Backup the environment :
`conda env export > environment.yml`

## Usage :
* Create a list of input  and target ranges in two text files (new line seperated). IPv6 addresses can be in the abbreviated format.
* Pass that file as an argument to ric :<br/>```ric -i input_ips.txt -t test_target.txt```

## Sample Output : 
```
$ ric -i input_ips.txt -t test_target.txt -v
### Device in use :  NVIDIA Tesla V100-PCIE-16GB
### Number Input IPs :  10 ; Target Size :  1000003
### Grid Dimensions of input init : ( 1 : 1024 )
### Grid Dimensions of target init : ( 977 : 1024 )
### Grid Dimensions of comparison :  (1, 32259) : (31, 31) )
3.1.1.0/24##Found in 3.1.0.0/16
1.1.1.1/32##Not_Found
10.0.1.1/32##Found in 10.0.0.0/8
10.65.1.1/32##Found in 10.64.0.0/10
192.168.255.255/32##Found in 192.168.0.0/16
10.0.1.0/24##Found in 10.0.0.0/8
172.25.24.0/23##Found in 172.16.0.0/12
192.168.0.0/32##Found in 192.168.0.0/16
255.255.255.255/32##Found in 255.255.0.0/16
0.0.0.0/0##Found in 192.168.0.0/16
###--- 2.6708738803863525 seconds ---
```
```
$ ric6 -i input_ips_v6.txt -t test_target_v6.txt -v
### Device in use :  NVIDIA Tesla V100-PCIE-16GB
### Number Input IPs :  5 ; Target Size :  4
### Grid Dimensions of input mask operations : ( 1 : 1024 )
### Grid Dimensions of target mask operations : ( 1 : 1024 )
### Grid Dimensions of comparison :  (1, 1) : (31, 31) )
2000:1:0000:0000:0000:0000:0000:/120 ##Found in 2000:0000:0000:0000:0000:0000:0000:/3
2000:0000:0000:0000:0000:0000:0000:1/128 ##Found in 2000:0000:0000:0000:0000:0000:0000:/3
ff01:0000:0000:0000:0000:0000:0000:1/128 ##Found in ff00:0000:0000:0000:0000:0000:0000:/8
ff00:0000:0000:0000:0000:0000:0000:2/127 ##Found in ff00:0000:0000:0000:0000:0000:0000:/8
0000:0000:0000:0000:0000:0000:0000:/8 ##Not_Found
###--- 6.19579553604126 seconds ---
```

## License

`rapid-ip-checker` is licensed under [Apache License 2.0](/LICENSE). 

## Code of Conduct

We take our [code of conduct](CODE_OF_CONDUCT.md) very seriously. Please abide 
by it.

## Contributing

Please read our [contributing guide](CONTRIBUTING.md) for details on how to 
contribute to our project.
