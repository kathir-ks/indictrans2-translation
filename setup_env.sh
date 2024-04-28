#bash script to setup the environment for the IndicTrans model

pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install torch 
pip install datasets transformers
pip install flax nltk

# git clone https://github.com/kathir-ks/setu-translate
mkdir flax_weights
# cd setu-translate/
# cd IndicTransTokenizer/
# pip install --editable ./

# Clone and install IndicTransTokenizer
git clone https://github.com/VarunGumma/IndicTransTokenizer
cd IndicTransTokenizer
pip install --editable ./
cd ..

sudo shutdown -r now #``Restart the VM to reflect the changes in the environment.