## Setting up Conda and ric for multiple users :

### Create a new user :
`sudo groupadd conda-users`

### Install Anaconda using .sh installation script as root. 
`sudo bash ./Anaconda3-2020.11-Linux-x86_64.sh`

### Set user permissions on the anaconda directories :
`sudo chgrp -R conda-users /usr/local/anaconda3/`

### Set permissions for the directories where the scripts and programs are placed :
`sudo chown -R :conda-users /usr/local/share/conda/`

### Add a user to the conda-users group :
`sudo usermod -a -G conda-users user_name`
