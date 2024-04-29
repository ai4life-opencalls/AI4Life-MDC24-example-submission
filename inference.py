"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-algorithm | gzip -c > example-algorithm.tar.gz

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""

from pathlib import Path

import imageio.v3 as imageio
import numpy as np
import torch
import SimpleITK


# Name of the expected input and output folders. DO NOT CHANGE.
INPUT_PATH = Path("/input/images/")
OUTPUT_PATH = Path("/output/images/")

# Name of the input grand challenge interface. CHANGE depending on the dataset.
INPUT_INTERFACE = "image-stack-structured-noise"
# Name of the output grand challenge interface
OUTPUT_INTERFACE = "image-stack-denoised"

# Path to the resource containing YOUR model. See 'src/create_model.py' for an example.
MODEL_PATH = Path("resources/model.pth")


def show_torch_cuda_info():
    """Print cuda information, so it can be availiable in the container logs"""
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)
    print("\n")


def save_result_image(image_array: np.ndarray, result_path: Path):
    """Save the result denoised image.
    Be careful to save results only in .mha format!
    Otherwise, the container will fail"""
    print(f"Writing image to: {result_path}")
    mha_image = SimpleITK.GetImageFromArray(image_array)
    SimpleITK.WriteImage(mha_image, result_path, useCompression=True)


def read_image(image_path: Path) -> (torch.Tensor, np.ndarray):
    """Read input noisy image from image_path"""
    print(f"Reading image: {image_path}")
    input_array = imageio.imread(image_path)
    input_array = input_array.astype(np.float32)
    print(f"Loaded image shape: {input_array.shape}")
    original_shape = input_array.shape
    # For this example, we will flatten the samples and channels to predict images one by one
    input_array = input_array.reshape(
        (-1, input_array.shape[-2], input_array.shape[-1])
    )
    input_tensor = torch.from_numpy(input_array)
    print(f"Final input shape: {input_tensor.shape}")
    return input_tensor, original_shape


def main():
    show_torch_cuda_info()

    input_folder = INPUT_PATH / INPUT_INTERFACE
    output_folder = OUTPUT_PATH / OUTPUT_INTERFACE
    output_folder.mkdir(exist_ok=True, parents=True)

    # Find all images in the input folder
    input_files = sorted(input_folder.glob(f"*.tif*"))
    print(f"Found files: {input_files}")

    # Load the example model
    print(f"Loading model: {MODEL_PATH}")
    model = torch.jit.load(MODEL_PATH)

    for input_file in input_files:
        input_tensor, original_shape = read_image(input_file)

        print("Running inference...")
        result = []
        # Run inference one sample at a time
        for x in input_tensor:
            x = x.unsqueeze(0)
            output = model(x).squeeze(0).numpy()
            result.append(output)

        output = np.stack(result, axis=0)
        output = output.reshape(original_shape)
        print(f"Output shape: {output.shape}")

        output_path = output_folder / f"{input_file.stem}.mha"
        save_result_image(image_array=output, result_path=output_path)


if __name__ == "__main__":
    main()