export packs_path=~/miniconda3/envs/m2cf/lib/python3.11/site-packages

# msCLAP model codes
export msclap_path=$packs_path/msclap
export patch_path=../code_pacthes/clap
cp $patch_path/clap.py $msclap_path/models/clap.py
cp $patch_path/htsat.py $msclap_path/models/htsat.py
cp $patch_path/CLAPWrapper.py $msclap_path/CLAPWrapper.py

# AudioMAE model codes
export patch_path=../code_pacthes/audiomae
cp $patch_path/misc.py ../AudioMAE/util/misc.py
cp $patch_path/pos_embed.py ../AudioMAE/util/pos_embed.py
