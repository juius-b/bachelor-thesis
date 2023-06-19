# CXR8

CXR8 is the name of the folder containing all relevant data for the ChestX-Ray8 database.

## Relevant components

The following are components of the database that are relevant for creating our own dataset of chest scans.

- Data_Entry_2017_v2020.csv: This csv file associates image files with patient data. Most importantly, it contains the findings in each scan.
- test_list.txt: This text file contains the file names of the images that make up the test set.
- train_val_list.txt: This text file contains the file names of the images that make up both the train and validation set. We need to split this into 70/80 % for the training set and 10/80 % for the validation set.
- images/images/: This directory contains all the source images from the study. They are 1024 x 1024 pixels and are supposed to be grayscale. However not all of them are grayscale and this has to be ensure so converting to array is no problem later

## Necessary changes

- Data_Entry_2017_v2020.csv
    - Rename columns "Image index", "Finding labels" "Patient ID" to "file_name", "findings" and "patient_id", respectively
    - Keep only those three columns 
    - Create dummy variables from findings with "|" being the seperator
    - Drop the resulting "no finding" column
    - Save to own file
- images/images/
    - Ensure grayscale
    - Resize to 256 x 256 pixels 
    - Save to own directory
    
- train_val_list.txt
    - Split up into train_list.txt and val_list.txt on a patient level while preserving the mentioned ratio

## Pipeline

This produces the npz file containing the label and image arrays.

1. Do mentioned necessary changes
2. With own image data file and own image directory
3. Open image data file and make file_name the index
4. For each file name in a text file listing the file names of a subset (e.g. train_list.txt listing the files making up the test set)
   - Get labels from image data file and add to labels list as NumPy array
   - Open image and add to images list as NumPy array
5. Convert labels and images lissts to NumPy arrays
6. Save arrays to single npz file