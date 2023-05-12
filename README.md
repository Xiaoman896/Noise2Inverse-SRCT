![version](https://img.shields.io/badge/Version-v1.0-blue.svg?style=plastic)
![tensorflow](https://img.shields.io/badge/TensorFlow-v2.2.0-green.svg?style=plastic)
![license](https://img.shields.io/badge/license-CC_BY--NC-red.svg?style=plastic)

# Noise2Inverse for SR-CT
Noise2inverse for synchrotron radiation CT use based on UNet (tensorflow)

Paper: [Hendriksen, A. A., Pelt, D. M., & Batenburg, K. J. (2020). Noise2inverse: Self-supervised deep convolutional denoising for tomography. IEEE Transactions on Computational Imaging, 6, 1320-1335.](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9178467);

[![DOI](https://zenodo.org/badge/DOI/10.1109/TCI.2020.3019647.svg)](https://doi.org/10.1109/TCI.2020.3019647)

To give it a try:

* prepare your raw CT collection data (folder contains three dataset: /tomo /flats /darks)
* create your virtual environment and install dependencies: 
  ```
  1. Open a terminal or command prompt.
  2. Navigate to the directory where you want to create the virtual environment (usually the path contains your script).
  3. Run the following command to create a new virtual environment: /usr/bin/python3.6 -m venv venv
  4. Activate the virtual environment: source venv/bin/activate
  6. Once the virtual environment is activated, run the following command to install the dependencies from the provided requirements.txt file: 
     pip install --upgrade pip
     pip install -r requirements.txt
* run Step1-tomos-split.py, an example:
  ```
  python Step1-tomos-split.py -image_raw_path /staff/.../4Al/15m/ -image_save_path /staff/.../4Al/15m/split -m 2
* run Step2-Reconstruction-tofu.py, before that you need to change the virtual environment to ufoenv: 
  ```
  deactivate
  source /opt/ufoenv/bin/activate
  ```
  An example for running Step2-Reconstruction-tofu.py:
  ```
  python Step2-Reconstruction-tofu.py -raw_dir /staff/.../4Al/15m/split/section_1/ -flag_PhR False -Ring_removal True
  ```
  If you have m subsections needed to be reconstructed, remember to run Step2-Reconstruction-tofu.py for m times with raw_dir=section_i, i=1,2,..,m.
* run Step3-Dataset_preparation.py, before that you need to change the virtual environment to venv you created:
  ```
  deactivate
  source venv/bin/activate
  ```
  An example for running Step3-Dataset_preparation.py:
  ```
  python Step3-Dataset_preparation.py -sli_dir /staff/.../4Al/15m/split/ -dataset_dir Dataset -gv_255 True -m 2
  ```
  Now you should have the h5 file saved in Dataset.

* run Step4-N2I-main.py, an example:
  ```
  python Step4-N2I-main.py -h5fn 580train_N2I_scaffold_Ring_removal_size_512.h5
  ```
  Now you should have a folder /Output/<your dataset's name>, with 
* run Step5-N2I-infer.py, an example:
  ```
  python Step5-N2I-infer.py -mdfn 580train_N2I_scaffold_Ring_removal_size_512_ssim_output -mdl N2I-it03000.h5 -dsfn Dataset/ -h5fn 580train_N2I_scaffold_Ring_removal_size_512.h5
  ```
  
