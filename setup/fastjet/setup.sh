#!/bin/bash

# Remove printout from pushd and popd
pushd () {
    command pushd "$@" > /dev/null
}

popd () {
    command popd "$@" > /dev/null
}

# Setup for (external) fastjet and fj-contrib. We don't necessarily need fj-contrib but it might be useful in the future.

# Get the current directory (and save it)
dir0=$PWD

# Curl is installed in our conda environment, so we activate it.
# For the following line, see https://stackoverflow.com/questions/2559076/how-do-i-redirect-output-to-a-variable-in-shell
read conda_str1 < <(conda info | grep -i 'base environment' | sed -e 's/[ ]*(.*$//g' | sed -e 's/[ ]*base environment :[ ]*//g')
conda_str2="/etc/profile.d/conda.sh"
conda_str="$conda_str1$conda_str2"
source "$conda_str"
conda activate ml4p

# Now we get the tar files and unpack them.
tfile1="fastjet-3.3.4.tar.gz"
tfile2="fjcontrib-1.045.tar.gz"

dir1="fastjet-src"
dir2="fj-contrib-src"

curl -O "http://fastjet.fr/repo/$tfile1"
wget "http://fastjet.hepforge.org/contrib/downloads/$tfile2"

mkdir -p $dir1
mkdir -p $dir2

tar zxvf $tfile1 -C $dir1 --strip-components=1
tar zxvf $tfile2 -C $dir2 --strip-components=1

#Now we have to install fastjet.

# Make the installation directory.
fj_install="fastjet-install"
mkdir -p fj_install

# Install fastjet.
pushd $dir1
    (./configure --prefix=$dir0/${fj_install} --enable-pyext=yes)
    [[ "$?" != 0 ]] && popd; 

    (make -j2)
    [[ "$?" != 0 ]] && popd; 

    (make install)
    [[ "$?" != 0 ]] && popd; 
popd

# Add to the library path and Python path. Not sure if Python path is really necessary here.
export LD_LIBRARY_PATH=${dir0}/${fj_install}/lib:$LD_LIBRARY_PATH
export PYTHONPATH=${dir0}/${fj_install}/lib/python3.8/site-packages:$PYTHONPATH

# To ensure that our conda env always sees the fastjet Python library, we add a .pth file.
# See https://stackoverflow.com/questions/37006114/anaconda-permanently-include-external-packages-like-in-pythonpath
#conda-develop ${dir0}/${fj_install}/lib/python3.8/site-packages # alternative if we have conda-buildar
path_file='fastjet.pth'
cat > ${conda_str1}/envs/ml4p/lib/python3.8/site-packages/${path_file} << EOF1
${dir0}/${fj_install}/lib/python3.8/site-packages
EOF1

# Install fj-contrib.
fjc_install="fjcontrib-install"
mkdir -p fjc_install

pushd $dir2
    flags="\"-O3 -Wall -g -fPIC -I ${dir0}${fj_install}/include"
    prefix="${dir0}/${fjc_install}"
    fj_config="${dir0}/${fj_install}/bin/fastjet-config"
    (./configure --prefix=$prefix --fastjet-config=$fj_config CXXFLAGS=$flags)
    [[ "$?" != 0 ]] && popd;
    
    (make -j2)
    [[ "$?" != 0 ]] && popd;
    
    (make install)
    [[ "$?" != 0 ]] && popd;     
popd

