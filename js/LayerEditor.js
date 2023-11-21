import { SideBar } from "./SideBar.js";
import { api } from "./api.js";
import { app } from "./app.js";
import { runONNX } from "./onnx.js";
import {
  showImageEditor,
  point_label,
  imageUrl,
  imageContainerSize,
  imagePrompts,
  targetNode,
  imageSize,
  selectedLayer,
  imagePromptsMulti,
  embeddings,
  embeddingID,
  alertDialog,
  allImagePrompts,
  boxesMulti,
} from "./state.js";
import { van } from "./van.js";
import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { PoseLandmarker, FaceLandmarker, FilesetResolver } = vision;
const { button, div, img, canvas, span } = van.tags;

let throttle = false;
const positivePrompt = van.state(true);
const enableBackgroundRemover = van.state(true);
const isMobileDevice = () => {
  return window.screen.width < 768;
};

// Auto segmentation
const filesetResolver = await FilesetResolver.forVisionTasks(
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
);
const faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
  baseOptions: {
    modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
    delegate: "GPU",
  },
  // outputFaceBlendshapes: true,
  runningMode: "IMAGE",
  numFaces: 1,
});
const poseLandmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
  baseOptions: {
    modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`,
    delegate: "GPU",
  },
  runningMode: "IMAGE",
  numPoses: 1,
});
const layerMapping = {
  L_eye: {
    useMiddle: false,
    positiveOffsetX: 0,
    positiveOffsetY: 0,
    negativeOffsetX: 0,
    negativeOffsetY: 0,
    positiveScale: 0,
    negativeScale: 0.5,
    indices: FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
  },
  R_eye: {
    useMiddle: false,
    positiveOffsetX: 0,
    positiveOffsetY: 0,
    negativeOffsetX: 0,
    negativeOffsetY: 0,
    positiveScale: 0,
    negativeScale: 0.5,
    indices: FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
  },
  L_iris: {
    useMiddle: false,
    positiveOffsetX: 0,
    positiveOffsetY: 0,
    negativeOffsetX: 0,
    negativeOffsetY: 0,
    positiveScale: -0.2,
    negativeScale: 0.5,
    indices: FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
  },
  R_iris: {
    useMiddle: false,
    positiveOffsetX: 0,
    positiveOffsetY: 0,
    negativeOffsetX: 0,
    negativeOffsetY: 0,
    positiveScale: -0.2,
    negativeScale: 0.5,
    indices: FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
  },
  face: {
    useMiddle: false,
    positiveOffsetX: 0,
    positiveOffsetY: 40,
    negativeOffsetX: 0,
    negativeOffsetY: 60,
    positiveScale: 0.2,
    negativeScale: 0.6,
    indices: FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
  },
  mouth: {
    useMiddle: true,
    positiveOffsetX: 0,
    positiveOffsetY: 0,
    negativeOffsetX: 0,
    negativeOffsetY: 0,
    positiveScale: 0,
    negativeScale: 0.5,
    indices: FaceLandmarker.FACE_LANDMARKS_LIPS,
  },
  mouth_in: {
    useMiddle: true,
    positiveOffsetX: 0,
    positiveOffsetY: 0,
    negativeOffsetX: 0,
    negativeOffsetY: 0,
    positiveScale: 0,
    negativeScale: 0.5,
    indices: FaceLandmarker.FACE_LANDMARKS_LIPS,
  },
};

export const segmented = van.state(false);

export async function autoSegment() {
  const image = document.getElementById("image");
  const landmarks = faceLandmarker.detect(image).faceLandmarks[0];

  Object.entries(layerMapping).forEach(([key, value]) => {
    imagePromptsMulti.val[key] = [];
  });

  Object.entries(layerMapping).forEach(([key, value]) => {
    const positivePoints = [];
    const middlePoints = [];
    const negativePoints = [];

    // Positive points
    for (const { start, end } of value.indices) {
      const startPoint = landmarks[start];
      // const endPoint = landmarks[end];

      const startX = startPoint.x * imageSize.val.width;
      const startY = startPoint.y * imageSize.val.height;

      // const endX = endPoint.x * imageSize.val.width;
      // const endY = endPoint.y * imageSize.val.height;

      if (middlePoints.length === 0) {
        middlePoints.push({ x: startX, y: startY, label: 1 });
        // middlePoints.push({ x: endX, y: endY, label: 1 });
      } else {
        middlePoints[0].x += startX;
        middlePoints[0].y += startY;
        // middlePoints[1].x += endX;
        // middlePoints[1].y += endY;
      }
      positivePoints.push({ x: startX, y: startY, label: 1 });
      // positivePoints.push({ x: endX, y: endY, label: 1 });

      // imagePrompts.val = [...imagePrompts.val, { x, y, label: 1 }];
    }

    // Middle points
    const len = value.indices.length;
    middlePoints[0].x /= len;
    middlePoints[0].y /= len;
    // middlePoints[1].x /= len;
    // middlePoints[1].y /= len;

    if (value.useMiddle) {
      imagePromptsMulti.val[key] = [
        ...imagePromptsMulti.val[key],
        ...middlePoints,
      ];
    } else {
      // Negative points
      for (const [i, { start, end }] of value.indices.entries()) {
        const startPoint = landmarks[start];
        // const endPoint = landmarks[end];

        const startX = startPoint.x * imageSize.val.width;
        const startY = startPoint.y * imageSize.val.height;

        // const endX = endPoint.x * imageSize.val.width;
        // const endY = endPoint.y * imageSize.val.height;

        const middlePoint = middlePoints[0];
        const directionVector = {
          x: middlePoint.x - startX,
          y: middlePoint.y - startY,
        };
        const directionVectorLength = Math.sqrt(
          directionVector.x * directionVector.x +
            directionVector.y * directionVector.y
        );
        const negativePointDistance =
          value.negativeScale * directionVectorLength;
        const negativePoint = {
          x:
            startX -
            (negativePointDistance * directionVector.x) /
              directionVectorLength -
            value.negativeOffsetX,
          y:
            startY -
            (negativePointDistance * directionVector.y) /
              directionVectorLength -
            value.negativeOffsetY,
          label: 0,
        };

        const positivePointDistance =
          value.positiveScale * directionVectorLength;
        positivePoints[i] = {
          x:
            positivePoints[i].x -
            (positivePointDistance * directionVector.x) /
              directionVectorLength -
            value.positiveOffsetX,
          y:
            positivePoints[i].y -
            (positivePointDistance * directionVector.y) /
              directionVectorLength -
            value.positiveOffsetY,
          label: 1,
        };

        negativePoints.push(negativePoint);
      }
      imagePromptsMulti.val[key] = [
        ...imagePromptsMulti.val[key],
        ...positivePoints,
        ...negativePoints,
      ];
      // Find bounding box of positive points
      const box = {
        x1: Math.min(...negativePoints.map((x) => x.x)),
        y1: Math.min(...negativePoints.map((x) => x.y)),
        x2: Math.max(...negativePoints.map((x) => x.x)),
        y2: Math.max(...negativePoints.map((x) => x.y)),
      };
      boxesMulti.val[key] = box;
    }
  });
  const poseLandmarks = poseLandmarker.detect(image).landmarks[0];
  const breathX =
    ((poseLandmarks[11].x + poseLandmarks[12].x) / 2) * imageSize.val.width;
  const breathY =
    ((poseLandmarks[11].y + poseLandmarks[12].y) / 2) * imageSize.val.height;
  imagePromptsMulti.val["breath"] = [{ x: breathX, y: breathY, label: 1 }];
  imagePrompts.val = imagePromptsMulti.val[selectedLayer.val];
  segmented.val = true;
  console.log("Done");
}

export function setRemoveBackgroundNode() {
  const rmBgNodes = app.graph.findNodesByType(
    "Image Rembg (Remove Background)"
  );
  if (!rmBgNodes?.length) {
    alertDialog.val = {
      text: "Remove background node not found. Please ensure the workflow is correct.",
      time: 5000,
    };
    return;
  }
  rmBgNodes.forEach((node) => {
    // node is bypassed if mode is 4
    node.mode = enableBackgroundRemover.val ? 0 : 4;
  });
}

export function updateImagePrompts() {
  if (selectedLayer.val !== "" && selectedLayer.val !== undefined) {
    imagePromptsMulti.val = {
      ...imagePromptsMulti.val,
      [selectedLayer.val]: imagePrompts.val,
    };

    targetNode.val.widgets.find((x) => x.name === "image_prompts_json").value =
      JSON.stringify(imagePromptsMulti.val);

    // const canvas = document.getElementById("mask-canvas");
    // const base64Image = canvas.toDataURL();
    // api.fetchApi("/segments", {
    //   method: "POST",
    //   body: JSON.stringify({
    //     name: embeddingID.val,
    //     segments: {
    //       [selectedLayer.val]: base64Image,
    //     },
    //   }),
    // });
  } else {
    targetNode.val.widgets.find((x) => x.name === "image_prompts_json").value =
      JSON.stringify(imagePrompts.val);
  }
  targetNode.val.graph.change();
}

export async function uploadSegments() {
  const emptyLayers = [];
  Object.entries(imagePromptsMulti.val).forEach(([key, value]) => {
    if (value.length === 0) {
      emptyLayers.push(key);
    }
  });
  if (emptyLayers.length > 0) {
    alertDialog.val = {
      text: "The following layers have no segments: " + emptyLayers.join(", "),
      time: 5000,
    };
    return false;
  }

  const segments = {};
  for (const [layer, prompts] of Object.entries(imagePromptsMulti.val)) {
    await drawSegment(getClicks(prompts));
    const canvas = document.getElementById("mask-canvas");
    const base64Image = canvas.toDataURL();
    segments[layer] = base64Image;
  }
  await api.fetchApi("/segments", {
    method: "POST",
    body: JSON.stringify({
      name: embeddingID.val,
      segments,
    }),
  });
  return true;
}

async function handleClick(e) {
  const rect = e.target.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const relativeX = Math.trunc(
    ((x / e.target.offsetWidth) * imageSize.val.width) / imageSize.val.imgScale
  );
  const relativeY = Math.trunc(
    ((y / e.target.offsetHeight) * imageSize.val.height) /
      imageSize.val.imgScale
  );

  let label;
  if (isMobileDevice()) {
    label = positivePrompt.val ? 1 : 0;
  } else {
    label = e.isRight ? 0 : 1;
  }

  imagePrompts.val = [
    ...imagePrompts.val,
    { x: relativeX, y: relativeY, label },
  ];
  await drawSegment(getClicks());
  updateImagePrompts();
}

async function handlePointClick(e, point) {
  e.preventDefault();
  imagePrompts.val = imagePrompts.val.filter(
    (x) => !(x.x === point.x && x.y === point.y)
  );
  await drawSegment(getClicks());
  updateImagePrompts();
}

function handleImageSize(image) {
  // Input images to SAM must be resized so the longest side is 1024
  const documentHeight = document.documentElement.clientHeight;
  const LONG_SIDE_LENGTH = 1024;
  let w = image.naturalWidth;
  let h = image.naturalHeight;
  const samScale = LONG_SIDE_LENGTH / Math.max(h, w);
  const imgScale = documentHeight / Math.max(h, w);
  return { height: h, width: w, samScale, imgScale };
}

export function getClicks(prompts) {
  return (prompts || imagePrompts.val).map((point) => ({
    x: point.x,
    y: point.y,
    clickType: point.label,
  }));
}

export async function drawSegment(clicks) {
  const canvas = document.getElementById("mask-canvas");
  const ctx = canvas.getContext("2d");
  if (clicks.length === 0) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  if (embeddings.val) {
    const box = boxesMulti.val[selectedLayer.val];
    const mask = await runONNX(clicks, embeddings.val, box);
    if (mask) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(mask, 0, 0);
      if (box) {
        ctx.strokeStyle = "green";
        ctx.lineWidth = 5;
        ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
      }
    }
  }
}

export function LayerEditor() {
  let realTimeSegment = true;

  const showSidebar = van.state(true);

  document.addEventListener("keydown", (e) => {
    if (showImageEditor.val && e.code === "Tab") {
      e.preventDefault();
      realTimeSegment = !realTimeSegment;
      if (!realTimeSegment) {
        drawSegment(getClicks());
      }
    }
  });

  return div(
    {
      class: () =>
        "absolute flex bg-gray-900 bg-opacity-50 top-0 w-full h-full pointer-events-auto z-[1000] " +
        (showImageEditor.val ? "" : "hidden"),
    },
    button(
      {
        class: () =>
          "btn btn-neutral flex flex-row normal-case absolute mt-4 rounded-md left-2 top-0 z-[200] w-fit",
        onclick: async () => {
          console.log("close");
          showImageEditor.val = false;
          await uploadSegments();

          const isEqual = allImagePrompts.val.map(
            (x) =>
              JSON.stringify(imagePromptsMulti.val) === JSON.stringify(x.prompt)
          );
          if (!isEqual.includes(true))
            allImagePrompts.val = [
              ...allImagePrompts.val,
              {
                version: "v" + allImagePrompts.val.length,
                prompt: imagePromptsMulti.val,
              },
            ];

          // api.fetchApi("/segments_order", {
          //   method: "POST",
          //   body: JSON.stringify({
          //     name: embeddingID.val,
          //     order: Object.keys(imagePromptsMulti.val),
          //   }),
          // });
        },
      },
      span({
        class: "iconify text-lg",
        "data-icon": "ic:baseline-arrow-back",
        "data-inline": "false",
      }),
      div("Back")
    ),
    button(
      {
        class: () =>
          "btn btn-neutral flex flex-row normal-case absolute mt-4 rounded-md left-28 top-0 z-[200] w-fit",
        onclick: () => (showSidebar.val = !showSidebar.val),
      },
      div(() => (showSidebar.val ? "Hide UI" : "Show UI"))
    ),
    button(
      {
        class: () =>
          "btn btn-neutral flex flex-row normal-case absolute mt-4 rounded-md left-52 top-0 z-[200] w-fit",
        onclick: () => {
          enableBackgroundRemover.val = !enableBackgroundRemover.val;
          setRemoveBackgroundNode();
        },
      },
      () =>
        enableBackgroundRemover.val
          ? "Background Remover On"
          : "Background Remover Off"
    ),
    button(
      {
        class: () =>
          `btn btn-neutral flex flex-row normal-case absolute mt-4 rounded-md left-52 top-0 z-[200] w-fit ${
            isMobileDevice() ? "" : "hidden"
          }`,
        onclick: () => (positivePrompt.val = !positivePrompt.val),
      },
      div(() => (positivePrompt.val ? "Positive" : "Negative"))
    ),
    div(
      {
        class:
          "hidden w-full justify-center absolute top-0 left-0 right-0 items-center",
      },
      button(
        {
          class: () => " px-4 py-2 rounded-md left-0 top-0 z-[200] ",
          onclick: () => {
            point_label.val = 1;
          },
        },
        "Positive"
      ),
      button(
        {
          class: () => " px-4 py-2 rounded-md left-0 top-0 z-[200] ",
          onclick: () => {
            point_label.val = 0;
          },
        },
        "Negative"
      )
    ),
    div(
      {
        class: "flex items-center justify-center w-full h-full",
        id: "image-container",
      },
      img({
        id: "image",
        class:
          "fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2",
        src: imageUrl,
        onload: async (e) => {
          imageSize.val = handleImageSize(e.target);

          document.getElementById("image-container").style.scale =
            imageSize.val.imgScale;

          imageContainerSize.val = {
            width: e.target.offsetWidth,
            height: e.target.offsetHeight,
          };

          const canvas = document.getElementById("mask-canvas");
          canvas.width = e.target.naturalWidth;
          canvas.height = e.target.naturalHeight;
        },
        oncontextmenu: async (e) => {
          e.preventDefault();
          e.isRight = true;
          await handleClick(e);
        },
        onclick: async (e) => {
          await handleClick(e);
        },
        onmouseleave: (e) => {
          drawSegment(getClicks());
        },
        onmousemove: (e) => {
          if (!throttle) {
            throttle = true;
            setTimeout(() => {
              throttle = false;

              if (embeddings.val && realTimeSegment) {
                const rect = e.target.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const relativeX = Math.trunc(
                  ((x / e.target.offsetWidth) * imageSize.val.width) /
                    imageSize.val.imgScale
                );
                const relativeY = Math.trunc(
                  ((y / e.target.offsetHeight) * imageSize.val.height) /
                    imageSize.val.imgScale
                );

                const clicks = [
                  ...getClicks(),
                  { x: relativeX, y: relativeY, clickType: 1 },
                ];
                drawSegment(clicks);
              }
            }, 10);
          }
        },
      }),
      () =>
        canvas({
          class:
            "pointer-events-none fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 opacity-80",
          style: () =>
            `width: ${imageContainerSize.val.width}px; height: ${imageContainerSize.val.height}px;`,
          id: "mask-canvas",
        }),
      () =>
        div(
          {
            class: "absolute w-full h-full pointer-events-none",
            style: () =>
              `width: ${imageContainerSize.val.width}px; height: ${imageContainerSize.val.height}px;`,
          },
          ...imagePrompts.val?.map((point) => {
            return button({
              style: () =>
                `left: ${
                  (point.x / imageSize.val.width) * imageContainerSize.val.width
                }px; top: ${
                  (point.y / imageSize.val.height) *
                  imageContainerSize.val.height
                }px; transform: translate(-50%, -50%);`,
              class: () =>
                `absolute w-3 h-3  p-0 rounded-full pointer-events-auto ${
                  point.label === 1 ? "bg-green-500" : "bg-red-500"
                }`,

              oncontextmenu: async (e) => {
                await handlePointClick(e, point);
              },
              onclick: async (e) => {
                await handlePointClick(e, point);
              },
            });
          })
        )
    ),
    () => (showSidebar.val ? SideBar() : div())
  );
}
