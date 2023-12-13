"""download toy dataset and convert to webdataset tarfile format"""
import os
import io
import sys
import argparse
import torch
import torchvision
import webdataset as wds

from PIL import Image


def download_dataset(data_dir, split):
    dataset = torchvision.datasets.MNIST(root=data_dir, train=(split == "train"), download=True)
    sink = wds.TarWriter(os.path.join(data_dir, f"mnist-{split}.tar"))
    
    for index, (image, label) in enumerate(dataset):
        # Print progress
        if index % 1000 == 0:
            print(f"{index:6d}", end="\r", flush=True)
        
        # Save image to bytes
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        image_bytes = buf.getvalue()

        # Prepare label as a text string
        label_str = str(label)

        # Write to tar file
        sink.write({
            "__key__": f"sample{index:06d}",
            "png": image_bytes,
            "txt": label_str,
        })

    sink.close()


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data", help="path to desired dataset location")
    args = parser.parse_args(args)
    return args


def main(args):
    args = parse_args(args)
    download_dataset(args.data_dir, "train")
    download_dataset(args.data_dir, "val")


if __name__ == "__main__":
    main(sys.argv[1:])
