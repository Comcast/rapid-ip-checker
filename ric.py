#/usr/local/anaconda3/envs/ric-rapids-0.19/bin/python3.7

####
## Copyright 2021 Comcast Cable Communications Management, LLC
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##
## SPDX-License-Identifier: Apache-2.0
####

from numba import cuda
import numpy as np
import time, sys
import operator
import argparse
import cudf

start_time= time.time()
parser = argparse.ArgumentParser(description='A Cuda Python based program to check if a list of input IPs is within a list of target networks.')
parser.add_argument('-i', '--input', action='store', dest='input_file', help='Specify Input list file')
parser.add_argument('-t', '--target', action='store', dest='target_file', help='Specify Target list file')
parser.add_argument('-v', '--verbose', action='store_true', dest='verbosity', help='Verbose Output')
if len(sys.argv) < 2:
    parser.print_usage()
    sys.exit(1)
args = parser.parse_args()

vprint = print if args.verbosity else lambda *a, **k: None

def load_array(filename):
  ## Gives a df like this :
  ##    0  1  2     3
  ## 0  1  2  3  4/24
  ## 1  5  6  7  8/25
  ## The decimal param  is passed to tell pandas not to interpret "." as decimal
  gdf = cudf.read_csv(filename, sep='.', decimal=",",  header=None, encoding="utf-8-sig")
  original_frame = gdf
  ## This section is to split the IPs last octet from the mask
  new_cols = gdf["3"].str.split("/", n = 1, expand = True)
  gdf.drop(columns =["3"], inplace = True)
  gdf[3]=new_cols[0]
  gdf[4]=new_cols[1].fillna(32)
  ## Convert the values in the frame to ints, and then into a numpy array
  gdf2=gdf.astype('int').as_matrix()
  ## Also return the number of IPs (number of rows)
  rows = gdf.shape[0]
  return gdf2 , rows

@cuda.jit
def IP_array_to_ints(ips, net_dec, mask):
  thread_pos = cuda.grid(1)
  net_dec[thread_pos] = (256**3)*ips[thread_pos][0]+(256**2)*ips[thread_pos][1]+(256)*ips[thread_pos][2]+ips[thread_pos][3]
  mask[thread_pos] = ips[thread_pos][4]


@cuda.jit
def compare_IP_to_IP(input_net_dec, input_mask, target_net_dec, target_mask, res_array):
  target_iterator, input_iterator = cuda.grid(2)
  if input_iterator < input_net_dec.size and target_iterator < target_net_dec.size :
    lower_cap = operator.ge((target_net_dec[target_iterator] + ((2**(32-target_mask[target_iterator]))-1)), input_net_dec[input_iterator])
    upper_cap = operator.ge((input_net_dec[input_iterator] + ((2**(32-input_mask[input_iterator]))-1)), target_net_dec[target_iterator])
    ## Checking if the networks overlap
    if ( lower_cap and upper_cap):
      res_array[input_iterator] =  target_iterator + 1

def reconstructed_ip(ip_array):
  return(str(ip_array[0]) + '.' + str(ip_array[1]) + '.' + str(ip_array[2]) + '.' + str(ip_array[3]) + '/' + str(ip_array[4]))

vprint("### Device in use : ", cuda.get_current_device().name.decode())

## Read the files and return them as numpy arrays and their size
input_ips, input_size =  load_array(args.input_file)
target_ips, target_size = load_array(args.target_file)

## Initialize all other arrays
input_net_dec = np.zeros(input_size, np.int64)
target_net_dec = np.zeros(target_size, np.int64)
input_mask = np.zeros(input_size, np.int64)
target_mask = np.zeros(target_size, np.int64)
res_array = np.zeros(input_size, np.int64)

## Convert Input Array (four point decimal) to an Integer Array :
blockdim = 1024  ## Blockdimension, that is number of threads is an upper limit.
griddim = (input_size + (blockdim - 1)) // blockdim
vprint("### Input Size : " , input_size, "; Target Size : " , target_size )
vprint("### Grid Dimensions of input init : (" , griddim, ":" , blockdim, ")")
  ## This is where net_dec and mask gets initialized (size declared)
net_dec = input_net_dec
mask = input_mask
IP_array_to_ints[griddim,blockdim](input_ips, net_dec, mask)
input_net_dec = net_dec
input_mask = mask

## Convert Target Array (four point decimal) to an Integer Array :
blockdim = 1024
griddim = (target_size + (blockdim - 1)) // blockdim
net_dec = target_net_dec
mask = target_mask
vprint("### Grid Dimensions of target init : (" , griddim, ":" , blockdim, ")")
IP_array_to_ints[griddim,blockdim](target_ips, net_dec, mask)

## Compare the two arrays. Number of threads allowed : 1024. Sq root of 1024 is 32
blockdim = (31, 31)
griddim = ( (target_size + (blockdim[1] - 1)) // blockdim[1], (input_size + (blockdim[0] - 1)) // blockdim[0])
vprint("### Grid Dimensions of comparison : (" , griddim, ":" , blockdim, ")")
compare_IP_to_IP[griddim, blockdim](input_net_dec, input_mask, net_dec, mask, res_array)

## Code to print results
for i in  range(res_array.size) :
  if (res_array[i] != 0) :
    print(reconstructed_ip(input_ips[i]), '##Found in ', reconstructed_ip(target_ips[res_array[i]-1]), sep = '')
  else :
    print(reconstructed_ip(input_ips[i]), '##Not_Found', sep = '')

vprint("###--- %s seconds ---" % (time.time() - start_time))
