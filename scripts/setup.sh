# cd back to main directory
cd ..

# create conda environment
conda create -n m2cf python=3.11
# make sure conda is correctly initialized
eval "$(conda shell.bash hook)"
# activate created environment
conda activate m2cf
# see if the environment is correct
conda env list

# install necessary packages
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt

# install minLoRA
git clone https://github.com/cccntu/minLoRA.git
pip install -e minLoRA/

# install AuioMAE
git clone https://github.com/facebookresearch/AudioMAE.git

# adapt the codes for environment compatibility and our needs
cd scripts/
if [ -d ~/miniconda3 ]; then
    # use miniconda
    chmod +x code_adapt_miniconda.sh
    ./code_adapt_miniconda.sh
else
    # Use anaconda
    chmod +x code_adapt_anaconda.sh
    ./code_adapt_anaconda.sh
fi
cd ..