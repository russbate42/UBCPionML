# Getting our trial data from some fixed locations (for now.)
user=jaofferm

type=$1 #this will indicate the kind of dataset we want to download

if [[ "$type" == "jet" ]]
then
    scp -r ${user}@lxplus.cern.ch:/eos/user/j/jaofferm/atlas-calo-ml/data/jet/* ./
elif [[ "$type" == "pion" ]]
then
    scp ${user}@lxplus.cern.ch:/eos/user/m/mswiatlo/images/v7/*.root ./
else
    echo "Type ${type} not understood."
fi