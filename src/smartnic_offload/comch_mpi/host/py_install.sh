export OMPI_HOME=/home/y-jinbo/opt/urom_stack/ompi
export PATH=$OMPI_HOME/bin:$PATH
export CC=$OMPI_HOME/bin/mpicc
export CXX=$OMPI_HOME/bin/mpicxx
export LDSHARED="$CXX -shared"
export LD_LIBRARY_PATH=/usr/local/lib:$OMPI_HOME/lib:$OMPI_HOME/lib64:$LD_LIBRARY_PATH

rm -rf build *.so
USE_MARCH_NATIVE=1 USE_LTO=1 python3 setup.py build_ext --inplace
