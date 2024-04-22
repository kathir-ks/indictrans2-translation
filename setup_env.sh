#bash script to setup the environment for the IndicTrans model

pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install pyspark
pip install torch torchvision torchaudio
pip install datasets transformers

git clone https://github.com/kathir-ks/setu-translate

cd setu-translate/
cd IndicTransTokenizer/
pip install --editable ./

sudo shutdown -r now #``Restart the VM to reflect the changes in the environment.