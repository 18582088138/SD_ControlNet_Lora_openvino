import argparse
import openvino.runtime as ov 
from openvino.tools.mo import convert_model
from openvino._offline_transformations import apply_moc_transformations, compress_model_transformation

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')

    args.add_argument('-h', '--help', action = 'help',
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model_path', type = str, default = "", required = True,
                      help='Requirt OpenVINO FP32 IR model path.')
    return parser.parse_args()

args = parse_args()

# fp32_model_path = "INT8/unet-controlnet.xml"
# fp16_model_path = "INT8/unet-controlnet-fp16.xml"
fp32_model_path = args.model_path
fp16_model_path = fp32_model_path.replace("FP32", "FP16")

core = ov.Core()
print("Read FP32 OV Model ...")
ov_model = core.read_model(fp32_model_path)

print("Convert FP32 OV Model to FP16 OV Model...")
apply_moc_transformations(ov_model, cf=False)
compress_model_transformation(ov_model)

print(f"Serialize Converted FP16 Model as {fp16_model_path}")
ov.serialize(ov_model, fp16_model_path)

print("Done.")