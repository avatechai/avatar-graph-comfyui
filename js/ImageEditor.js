import {
  showImageEditor,
  point_label,
  imageUrl,
  imageContainerSize,
  imagePrompts,
  targetNode,
  imageSize,
  selectedLayer,
} from './state.js';
import { van } from './van.js';
const { button, iframe, div, img } = van.tags;

function updateImagePrompts() {
  targetNode.val.widgets.find((x) => x.name === 'image_prompts_json').value =
    JSON.stringify(imagePrompts.val);
  targetNode.val.graph.change();
}

function handleClick(e) {
  const rect = e.target.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const relativeX = Math.trunc(
    (x / e.target.offsetWidth) * imageSize.val.width,
  );
  const relativeY = Math.trunc(
    (y / e.target.offsetHeight) * imageSize.val.height,
  );

  imagePrompts.val = [
    ...imagePrompts.val,
    { x: relativeX, y: relativeY, label: e.isRight ? 0 : 1 },
  ];

  updateImagePrompts();
}

function handlePointClick(e, point) {
  e.preventDefault();
  imagePrompts.val = imagePrompts.val.filter(
    (x) => x.x !== point.x && x.y !== point.y,
  );
  updateImagePrompts();
}

export function ImageEditor() {
  return div(
    {
      class: () =>
        'absolute flex bg-gray-900 bg-opacity-50 top-0 w-full h-full pointer-events-auto  ' +
        (showImageEditor.val ? '' : 'hidden'),
    },
    button(
      {
        class: () => 'absolute px-4 py-2 rounded-md left-0 top-0 z-[200] ',
        onclick: () => {
          console.log('close');
          showImageEditor.val = false;
        },
      },
      'back',
    ),
    div(
      {
        class:
          'hidden w-full flex justify-center absolute top-0 left-0 right-0 items-center',
      },
      button(
        {
          class: () => ' px-4 py-2 rounded-md left-0 top-0 z-[200] ',
          onclick: () => {
            point_label.val = 1;
          },
        },
        'Positive',
      ),
      button(
        {
          class: () => ' px-4 py-2 rounded-md left-0 top-0 z-[200] ',
          onclick: () => {
            point_label.val = 0;
          },
        },
        'Negative',
      ),
    ),
    div(
      {
        class:
          'flex items-center justify-center absolute h-full left-0 right-0 bottom-0 top-0 mx-auto my-auto max-h-[1000px] ',
      },
      img({
        class: 'w-fit h-full',
        src: imageUrl,
        onload: (e) => {
          imageSize.val = {
            width: e.target.naturalWidth,
            height: e.target.naturalHeight,
          };

          imageContainerSize.val = {
            width: e.target.offsetWidth,
            height: e.target.offsetHeight,
          };
        },
        oncontextmenu: (e) => {
          e.preventDefault();
          e.isRight = true;
          handleClick(e);
        },
        onclick: (e) => {
          handleClick(e);
        },
      }),
      () => {
        return div(
          {
            class: 'absolute w-full h-full pointer-events-none',
            style: () =>
              `width: ${imageContainerSize.val.width}px; height: ${imageContainerSize.val.height}px;`,
          },
          ...imagePrompts.val.map((point) => {
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
                  point.label === 1 ? 'bg-green-500' : 'bg-red-500'
                }`,

              oncontextmenu: (e) => {
                handlePointClick(e, point);
              },
              onclick: (e) => {
                handlePointClick(e, point);
              },
            });
          }),
        );
      },
    ),
  );
}
