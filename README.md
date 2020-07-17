# DevanagariOCR
An End-End DL based model on Devanagari OCR

The pipeline consists two phases: Text Detection and Text Recognition. For Text detection, EAST model trained on ICDAR 2013 and ICDAR 2015 dataset is used which is later
fine tuned for devanagari language by changing the threshold values. For Text Recognition, CRNN model is used which is trained on IIITH Hindi [dataset](https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data)

The implementation of EAST: Efficient and Accurate Scene Text Detector is borrowed from this [repository.](https://github.com/kurapan/EAST)

## Requirements
Keras 2.0 or higher, and TensorFlow 1.0 or higher should be enough

## Data
The trained EAST model can be downloaded from [here.](https://drive.google.com/file/d/1Zf-hNC4XpxFLMPQoMBkIRZH2Cia15hCn/view?usp=sharing)
The trained CRNN model can be downloaded from [here.](https://drive.google.com/file/d/1JBS19RG73S6PfbV1CAdANbzqGB6id186/view?usp=sharing)


