## Steps for a user to access and run ric in a shared environment :

### (Optional) Set up a tmux session for a permanent environment :
`tmux new -t conda`

### Activate Anaconda :
`eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"`

### Activate our shared environemt :
`conda activate rapids-0.19`

### Create a bash alias for the program :
`alias ric="python /usr/local/share/ric/ric.py"; source ~/.bashrc ; alias > ~/.bash_aliases`

### Run the Program :
`ric -i input_ips.txt`

### Deactivate the shared environment when not required :
`conda deactivate`
