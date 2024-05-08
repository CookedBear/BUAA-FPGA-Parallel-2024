export TVM_HOME=/mnt/cgshare/tvm-09c
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/vta/python:${PYTHONPATH}
export PATH=/usr/local/gcc-9/bin:$PATH
export LD_LIBRARY_PATH=/mnt/cgshare/llvm9/lib
export MANPATH=/usr/local/gcc-9/share/man:$MANPATH
export PATH=$PATH:/mnt/cgshare/llvm9/bin
export VTA_PYNQ_RPC_HOST=192.168.134.123
export VTA_PYNQ_RPC_PORT=9941