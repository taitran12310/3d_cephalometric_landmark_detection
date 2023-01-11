import SimpleITK as sitk
import sys
import os
import time

image_scale = (72, 96, 96)
new_width = 96
if len(sys.argv) < 3:
    # sample [python convert.py "ceph1" "ceph1.nii.gz"]
    print("Usage: " + __file__ + " <input_directory> <output_file>") #output is .nii file
    sys.exit(1)
root_dir = "processed_data"
input_dir = os.path.join(root_dir, "images/train")
output_dir = os.path.join(root_dir, "output")

input = os.path.join(input_dir, sys.argv[1])
output_resize = os.path.join(output_dir, "72_"+ sys.argv[2])
output = os.path.join(output_dir, sys.argv[2])
print("Reading Dicom directory:", input)

reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(input)
reader.SetFileNames(dicom_names)

image3D = reader.Execute()
original_size = image3D.GetSize()
original_spacing = image3D.GetSpacing()
new_spacing = [(original_size[0] - 1) * original_spacing[0] / (new_width - 1)] * 3

image = sitk.Resample(
    image1=image3D,
    size=image_scale,
    transform=sitk.Transform(),
    interpolator=sitk.sitkLinear,
    outputOrigin=image3D.GetOrigin(),
    outputSpacing=new_spacing,
    outputDirection=image3D.GetDirection(),
    defaultPixelValue=0,
    outputPixelType=image3D.GetPixelID(),
)

size = image3D.GetSize()
print("Image size:", size[0], size[1], size[2])

print("Writing image:", output)

sitk.WriteImage(image, output_resize)
sitk.WriteImage(image3D, output)

sys.exit(0)