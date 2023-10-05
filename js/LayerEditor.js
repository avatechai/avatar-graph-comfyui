import { SideBar } from "./SideBar.js";
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
} from "./state.js";
import { van } from "./van.js";
const { button, div, img, canvas, span } = van.tags;

let throttle = false;

export function updateImagePrompts() {
  if (selectedLayer.val !== "" && selectedLayer.val !== undefined) {
    imagePromptsMulti.val = {
      ...imagePromptsMulti.val,
      [selectedLayer.val]: imagePrompts.val,
    };

    targetNode.val.widgets.find((x) => x.name === "image_prompts_json").value =
      JSON.stringify(imagePromptsMulti.val);
  } else {
    targetNode.val.widgets.find((x) => x.name === "image_prompts_json").value =
      JSON.stringify(imagePrompts.val);
  }
  targetNode.val.graph.change();
}

function handleClick(e) {
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

  imagePrompts.val = [
    ...imagePrompts.val,
    { x: relativeX, y: relativeY, label: e.isRight ? 0 : 1 },
  ];

  updateImagePrompts();
  drawSegment(getClicks());
}

function handlePointClick(e, point) {
  e.preventDefault();
  imagePrompts.val = imagePrompts.val.filter(
    (x) => !(x.x === point.x && x.y === point.y)
  );
  updateImagePrompts();
  drawSegment(getClicks());
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

export function getClicks() {
  return imagePrompts.val.map((point) => ({
    x: point.x,
    y: point.y,
    clickType: point.label,
  }));
}

export function drawSegment(clicks) {
  const canvas = document.getElementById("mask-canvas");
  const ctx = canvas.getContext("2d");
  if (clicks.length === 0) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  if (embeddings.val) {
    runONNX(clicks, embeddings.val).then((mask) => {
      if (mask) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(mask, 0, 0);
      }
    });
  }
}

export function LayerEditor() {
  let realTimeSegment = true;
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
        "absolute flex bg-gray-900 bg-opacity-50 top-0 w-full h-full pointer-events-auto " +
        (showImageEditor.val ? "" : "hidden"),
    },
    button(
      {
        class: () =>
          "btn btn-circle flex flex-row btn-ghost normal-case absolute p-0 rounded-md left-2 top-0 z-[200] w-fit",
        onclick: () => {
          console.log("close");
          showImageEditor.val = false;
        },
      },
      span({
        class: "iconify text-lg",
        "data-icon": "ic:baseline-arrow-back",
        "data-inline": "false",
      }),
      div("Back")
    ),
    div(
      {
        class:
          "hidden w-full flex justify-center absolute top-0 left-0 right-0 items-center",
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
        class:
          "fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2",
        src: imageUrl,
        onload: (e) => {
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
        oncontextmenu: (e) => {
          e.preventDefault();
          e.isRight = true;
          handleClick(e);
        },
        onclick: (e) => {
          handleClick(e);
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
      canvas({
        class:
          "pointer-events-none fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 opacity-80",
        id: "mask-canvas",
      }),
      () => {
        return div(
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

              oncontextmenu: (e) => {
                handlePointClick(e, point);
              },
              onclick: (e) => {
                handlePointClick(e, point);
              },
            });
          })
        );
      }
    ),
    SideBar()
  );
}
