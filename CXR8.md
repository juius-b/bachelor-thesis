# CXR8

CXR8 is the name of the folder containing all relevant data for the ChestX-Ray8 database.

## Relevant components

The following are components of the database that are relevant for creating our own dataset of chest scans.

- images/images/: Directory of all source images from the study
  - 1024 x 1024 pixels
  - almost exclusively grayscale
- chestmnist_split_info.csv: Csv file provided by the authors of the MedMNIST v2 paper associating images files (image_id) with the index in one of the splits
- chestmnist.npz: Npz file from the medmnist package that contains the labels for the images referenced in the split info file

## Resize requirements

- images/images/
    - Ensure grayscale
    - Resize to wanted size (e.g. 256)

## Pipeline

This produces the npz file containing the label and image arrays.


1. Read split info
2. Create dictionary for image arrays of splits
3. In parallel
   1. Get split, index and image_id from split info row
   2. Convert to grayscale if necessary
   3. Resize image_id.png
   4. Write image as array to correct array at provided index determined by dictionary and split
4. Read labels for each split from mnist npz file
5. Save own compressed npz file by combining own image arrays and the read labels