import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path=None):
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_builder_config() as config, \
            builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        #builder.max_workspace_size = 1 << 30  
        #config.max_workspace_size = 1 << 30 
        #config.set_flag(trt.BuilderFlag.FP16)

        print("Loading ONNX file from path {}...".format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        print("Completed parsing of ONNX file")
        print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
        engine = builder.build_engine(network, config)

        if engine_file_path is not None:
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())

        return engine

# Example usage:
onnx_file_path = '/home/016712345/yolov5/fasterrcnn_resnet50_fpn.onnx'
engine = build_engine(onnx_file_path, engine_file_path="fasterrcnn_resnet50_fpn.trt")

