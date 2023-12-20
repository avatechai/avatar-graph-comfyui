import "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js";
import npyjs from "https://esm.sh/npyjs";
import { imageSize } from "./state.js";
import { modelData, onnxMaskToImage } from "./onnx_helper.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

export let model = null;

// Initialize the ONNX model
export const initModel = async (modelType) => {
  try {
    if (!model) {
      model = await ort.InferenceSession.create(
        `${location.protocol}//${location.host}/sam_model?type=${modelType}`
      );
    }
  } catch (e) {
    console.log(e);
  }
};

export const loadNpyTensor = async (tensorFile, dType = "float32") => {
  let npLoader = new npyjs();
  const npArray = await npLoader.load(tensorFile);
  const tensor = new ort.Tensor(dType, npArray.data, npArray.shape);
  return tensor;
};

export const runONNX = async (clicks, tensor, box) => {
  // console.log('tensor', tensor);
  try {
    if (
      model === null ||
      clicks === null ||
      tensor === null ||
      imageSize.val === null
    )
      return;
    else {
      // Preapre the model input in the correct format for SAM.
      // The modelData function is from onnxModelAPI.tsx.
      const feeds = modelData({
        clicks,
        tensor,
        modelScale: imageSize.val,
        box,
      });
      if (feeds === undefined) return;
      // Run the SAM ONNX model with the feeds returned from modelData()
      const results = await model.run(feeds);
      const output = results[model.outputNames[0]];
      // The predicted mask returned from the ONNX model is an array which is
      // rendered as an HTML image using onnxMaskToImage() from maskUtils.tsx.
      return onnxMaskToImage(output.data, output.dims[2], output.dims[3]);
    }
  } catch (e) {
    console.log(e);
  }
};
