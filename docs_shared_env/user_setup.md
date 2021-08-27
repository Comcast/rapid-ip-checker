## Steps for a user to access and run ric in a shared environment :

### (Optional) Set up a tmux session for a permanent environment :
`tmux new -t conda`

### Activate Anaconda :
`eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"`

### Activate our shared environemt :
`conda activate rapids-0.19`

### Create a bash alias for the program :
```
alias ric="python ./ric.py"; source ~/.bashrc ; alias > ~/.bash_aliases
alias ric6="python ./ric6.py"; source ~/.bashrc ; alias > ~/.bash_aliases
```

### Run the Program :
For IPv4 addresses : `ric -i input_ipv4_file.txt -t target_ipv4_file.txt`<br />
For IPv6 addresses : `ric6 -i input_ipv6_file.txt -t target_ipv6_file.txt`

### Deactivate the shared environment when not required :
`conda deactivate`
