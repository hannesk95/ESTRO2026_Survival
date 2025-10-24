import os
import torch
from radiomics import featureextractor
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm

def extract_features(dataset_path):
    """
    Extracts radiomics features from 3D medical images and their corresponding segmentation masks.
    Saves the features for each patient in a separate .pt file.

    Args:
        dataset_path (str): Path to the dataset containing image and mask files.
                            Assumes images are named with '_image' and masks with '_mask'.
    """

    # ---------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------

    # lists of file paths
    image_paths = sorted(glob(os.path.join(dataset_path, "*image_preprocessed.nii.gz")))
    mask_paths = sorted(glob(os.path.join(dataset_path, "*label_preprocessed.nii.gz")))

    assert len(image_paths) == len(mask_paths), "Image and mask lists must have same length."

    # ---------------------------------------------------------------------
    # PyRadiomics feature extractor setup
    # ---------------------------------------------------------------------

    settings = {
        "resampledPixelSpacing": None,  # Do not resample
        "interpolator": "sitkBSpline",  # Irrelevant since no resampling
        "enableCExtensions": True,
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    print("Enabled feature classes:")
    for cls, features in extractor.enabledFeatures.items():
        print(f"  {cls}: {features}")

    # ---------------------------------------------------------------------
    # Extraction loop
    # ---------------------------------------------------------------------

    for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):

        try:

            print(img_path)
            print(mask_path)

            # Read image and mask
            image = sitk.ReadImage(img_path)
            mask = sitk.ReadImage(mask_path)

            # Extract features
            result = extractor.execute(image, mask)

            # Filter out diagnostic entries
            features_dict = {k: float(v) for k, v in result.items() if not k.startswith("diagnostics_")}

            # Save to individual .pt file       
            torch.save(features_dict, img_path.replace(".nii.gz", ".pt").replace("image_preprocessed", "features"))

        except:
            continue

if __name__ == "__main__":

    for dataset_path in ["/home/johannes/Data/SSD_2.0TB/Duschinger/data/survival/duschinger/final_dataset"]:
        
        extract_features(dataset_path)
