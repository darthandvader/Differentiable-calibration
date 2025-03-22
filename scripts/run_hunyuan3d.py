import argparse
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


def main(image_path, save_path):
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
    mesh = pipeline(image=image_path)[0]

    print("Saving demo file.")
    mesh.export(save_path)
    print(f"Done saving to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-image-path", type=str, default="submodules/hunyuan3d/assets/demo.png"
    )
    parser.add_argument("--output-obj-path", type=str, default="demo.obj")
    args = parser.parse_args()
    main(image_path=args.input_image_path, save_path=args.output_obj_path)
