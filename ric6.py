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
parser = argparse.ArgumentParser(description='A Cuda Python based program to check if a list of input IPs is within a list of target networks \n python ips_in_sel_check.py -i input_ips.txt -t sel.txt')
parser.add_argument('-i', '--input', action='store', dest='input_file', help='Specify Input list file')
parser.add_argument('-t', '--target', action='store', dest='target_file', help='Specify Target list file')
parser.add_argument('-v', '--verbose', action='store_true', dest='verbosity', help='Verbose Output')
if len(sys.argv) < 2:
    parser.print_usage()
    sys.exit(1)
args = parser.parse_args()

vprint = print if args.verbosity else lambda *a, **k: None

def load_array_v6(filename):
  df= cudf.read_csv(filename,  header=None, encoding="utf-8-sig", dtype="str")
  ## Unpacking an IPv6 address (On the CPU). Eg: 0::0/128 to 0:0:0:0:0:0:0:0/128
  for x in range(7):
   df.loc[((df["0"].str.contains("::") == True)  & (df["0"].str.count(':') < 8)), "0"] = df["0"].str.replace('::', ':0000::')
  df.loc[((df["0"].str.contains("::") == True)  & (df["0"].str.count(':') == 8)), "0"] = df["0"].str.replace('::', ':')
  ## Saving the original dataframes to print in results
  original_frame = df["0"].to_pandas()
  ## Convert the pandas dataframe into a cudf dataframe
  gdf = cudf.DataFrame()
  new_cols = df["0"].str.split("/", n = 1, expand = True)
  df=new_cols[0].str.split(":", n = 8, expand = True)
  gdf[0]=new_cols[1].fillna(128)
  ## Convert each split octet into its decimal equivalent (16 bits)
  for x in range(8):
   df[x] = df[x].str.htoi()
  gdf_ips=df.astype('int').to_pandas().to_numpy()
  gdf_mask=gdf.astype('int').to_pandas().to_numpy()
  rows = gdf.shape[0]
  return original_frame, gdf_ips, gdf_mask, rows

@cuda.jit
def compare_net_to_net_v6(input_net_dec, input_mask, target_net_dec, target_mask, res_array):
  target_iterator, input_iterator = cuda.grid(2)
  if input_iterator < (input_net_dec.size/8) and target_iterator < (target_net_dec.size/8) :
    lower_cap = 0
    upper_cap = 0
    overlap_octet = 0
    for x in range(8):
      lower_cap =  operator.ge((target_net_dec[target_iterator][x] + target_mask[target_iterator][x]), input_net_dec[input_iterator][x])
      upper_cap = operator.ge((input_net_dec[input_iterator][x] + input_mask[input_iterator][x]), target_net_dec[target_iterator][x])
      if (lower_cap  and upper_cap) :
        overlap_octet += 1
      else :
        break
    if (overlap_octet == 8):
      res_array[input_iterator] = target_iterator + 1

@cuda.jit
def mask_split_v6(input_array, generic_mask_split):
  thread_pos = cuda.grid(1)
  prev = 0
  for x in range(8):
    i = 16*(x+1) - input_array[thread_pos][0]
    if operator.gt(i, 0):
      generic_mask_split[thread_pos][x] = 2**(i - prev)-1
      prev += (i - prev)

vprint("### Device in use : ", cuda.get_current_device().name.decode())

## Read the files and return them as numpy arrays and their size
input_initial, input_ips, input_mask, input_size =  load_array_v6(args.input_file)
target_initial, target_ips, target_mask, target_size =  load_array_v6(args.target_file)

## Initialize some numpy arrays
res_array = np.zeros(input_size, np.int64)
input_mask_split = np.zeros((input_size,8), np.int64)
target_mask_split = np.zeros((target_size,8), np.int64)


blockdim = 1024  ## Blockdimension, that is number of threads is an upper limit.
  ## Blockdimension, that is number of threads is an upper limit.
griddim = (input_size + (blockdim - 1)) // blockdim
vprint("### Number Input IPs : " , input_size, "; Target Size : " , target_size )

## Split the 128 bit mask to return size in decimal, per octet
generic_mask_split = input_mask_split
if griddim_x > MAX_GRID_DIM_X or griddim_y > MAX_GRID_DIM_Y:
    raise ValueError(f"Input size or target size exceeds maximum grid dimensions: ({griddim_x}, {griddim_y})")
vprint("### Grid Dimensions of input mask operations : (" , griddim, ":" , blockdim, ")")
mask_split_v6[griddim,blockdim](input_mask, generic_mask_split)
input_mask_split = generic_mask_split

generic_mask_split = target_mask_split
griddim = (target_size + (blockdim - 1)) // blockdim
if griddim_x > MAX_GRID_DIM_X or griddim_y > MAX_GRID_DIM_Y:
    raise ValueError(f"Input size or target size exceeds maximum grid dimensions: ({griddim_x}, {griddim_y})")
vprint("### Grid Dimensions of target mask operations : (" , griddim, ":" , blockdim, ")")
mask_split_v6[griddim,blockdim](target_mask, generic_mask_split)
target_mask_split = generic_mask_split

## Compare the two arrays
MAX_GRID_DIM_X = 2147483647
MAX_GRID_DIM_Y = 65535

blockdim = (31, 31)
griddim_x = ((target_size + (blockdim[1] - 1)) // blockdim[1])
griddim_y = ((input_size + (blockdim[0] - 1)) // blockdim[0])

# Validate grid dimensions
if griddim_x > MAX_GRID_DIM_X or griddim_y > MAX_GRID_DIM_Y:
    raise ValueError(f"Input size or target size exceeds maximum grid dimensions: ({griddim_x}, {griddim_y})")

griddim = (griddim_x % MAX_GRID_DIM_Y, griddim_y % MAX_GRID_DIM_Y)
vprint("### Grid Dimensions of comparison : " , griddim, ":" , blockdim, ")")
compare_net_to_net_v6[griddim, blockdim](input_ips, input_mask_split, target_ips, target_mask_split, res_array)

for i in  range(res_array.size) :
  if (res_array[i] != 0) :
    print(input_initial[i], '##Found in', target_initial[res_array[i]-1])
  else:
    print(input_initial[i], '##Not_Found')

vprint("###--- %s seconds ---" % (time.time() - start_time))
