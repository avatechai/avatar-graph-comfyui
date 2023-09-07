
/** @typedef {import('../../../web/types/comfy.js').ComfyExtension} ComfyExtension*/
/** @typedef {import('../../../web/scripts/app.js').ComfyApp} ComfyApp*/
/** @typedef {import('../../../web/scripts/api.js').ComfyApi} API*/

import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';

/** @type {ComfyApp} */
const app = app

/** @type {API} */
const api = api

import * as _van from 'https://cdn.jsdelivr.net/gh/vanjs-org/van/public/van-1.1.3.min.js';
/** @type {import('./van-1.1.3.min.js').Van} */
const van = _van.default;

function addMenuHandler(nodeType, cb) {
  const getOpts = nodeType.prototype.getExtraMenuOptions;
  nodeType.prototype.getExtraMenuOptions = function () {
    const r = getOpts.apply(this, arguments);
    cb.apply(this, arguments);
    return r;
  };
}

function addExtraButton(nodeType, cb) {
  const getOpts = nodeType.prototype.createWidget;
  nodeType.prototype.createWidget = function () {
    const r = getOpts.apply(this, arguments);
    cb.apply(this, arguments);
    return r;
  };
}

const iframeSrc = van.state('https://editor.avatech.ai');
const showEditor = van.state(false);
const showImageEditor = van.state(false);
const imageUrl = van.state('');
const point_label = van.state(1);
const imageContainerSize = van.state({
  width: 0,
  height: 0,
});

/**
 * @typedef {Object} Point
 * @property {number} x - The x coordinate
 * @property {number} y - The y coordinate
 * @property {number} label - The label
 */

/** @type {import('./van-1.1.3.min.js').State<Point[]>} */
const imagePrompts = van.state([]);
let targetWidget;
let imageSize = { width: 0, height: 0 };

function openInAvatechEditor(url, fileName) {
  let editor = document.getElementById('avatech-editor-iframe');
  iframeSrc.val = url;
  showEditor.val = true;

  editor.contentWindow.postMessage(
    {
      key: 'key',
      value: fileName,
      method: 'store',
    },
    '*',
  );
}

app.registerExtension({
  name: 'Avatech.Avatar.BlendshapeEditor',
  init(app) {},

  async setup() {
    const graphCanvas = document.getElementById('graph-canvas');
    window.addEventListener('keydown', (event) => {
      console.log(event.key);

      if (event.key === 'b') {
        event.preventDefault();
        const currentGraph = app.graph.list_of_graphcanvas[0];
        if (currentGraph.selected_nodes.length !== 1) {
          Object.values(currentGraph.selected_nodes).forEach((targetNode) => {
            if (targetNode.mode === 4) targetNode.mode = 0;
            else targetNode.mode = 4;
          });
        } else {
          const targetNode = currentGraph.current_node;
          if (targetNode.mode === 4) targetNode.mode = 0;
          else targetNode.mode = 4;
        }
        app.graph.change();
      }

      if (event.key === 'Escape') {
        event.preventDefault();
        showImageEditor.val = false;
      }

      // if enter
      if (event.key === 'Enter') {
        event.preventDefault();
        document.getElementById('queue-button').click();
      }
    });

    const { button, iframe, div, img } = van.tags;

    const AvatarEditor = () => {
      return div(
        {
          class: 'w-full h-full',
        },
        button(
          {
            class: () =>
              'px-4 py-2 rounded-md absolute left-0 top-0 z-[100] pointer-events-auto ' +
              (showEditor.val ? '' : 'hidden'),
            onclick: () => {
              console.log('close');
              showEditor.val = false;
            },
          },
          'back',
        ),
        iframe({
          id: 'avatech-editor-iframe',
          title: 'avatech-editor-iframe',
          name: 'avatech-editor-iframe',
          class: () =>
            'w-full h-full pointer-events-auto ' +
            (showEditor.val ? '' : 'hidden'),
          src: iframeSrc,
        }),
      );
    };

    const ImageEditor = () => {
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
              'w-full flex justify-center absolute top-0 left-0 right-0 items-center',
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
              'flex items-center justify-center absolute h-full left-0 right-0 bottom-0 top-0 mx-auto my-auto max-h-96 ',
          },
          img({
            class: 'w-fit h-full',
            src: imageUrl,
            onload: (e) => {
              imageSize = {
                width: e.target.naturalWidth,
                height: e.target.naturalHeight,
              };

              imageContainerSize.val = {
                width: e.target.offsetWidth,
                height: e.target.offsetHeight,
              };
            },
            onclick: (e) => {
              const rect = e.target.getBoundingClientRect();
              const x = e.clientX - rect.left;
              const y = e.clientY - rect.top;
              const relativeX = Math.trunc(
                (x / e.target.offsetWidth) * imageSize.width,
              );
              const relativeY = Math.trunc(
                (y / e.target.offsetHeight) * imageSize.height,
              );

              imagePrompts.val = [
                ...imagePrompts.val,
                { x: relativeX, y: relativeY, label: point_label.val },
              ];

              // targetWidget.widgets.find((x) => x.name === 'x').value = relativeX;
              // targetWidget.widgets.find((x) => x.name === 'y').value = relativeY;
              targetWidget.widgets.find(
                (x) => x.name === 'image_prompts_json',
              ).value = JSON.stringify(imagePrompts.val);
              targetWidget.graph.change();
              // showImageEditor.val = false;
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
                return div(
                  {
                    class: 'absolute',
                    style: () =>
                      `left: ${
                        (point.x / imageSize.width) *
                        imageContainerSize.val.width
                      }px; top: ${
                        (point.y / imageSize.height) *
                        imageContainerSize.val.height
                      }px; transform: translate(-50%, -50%);`,
                  },
                  button({
                    class: () =>
                      `w-4 h-4 rounded-full pointer-events-auto ${
                        point.label === 1 ? 'bg-green-500' : 'bg-red-500'
                      }`,
                    onclick: (e) => {
                      e.preventDefault();
                      imagePrompts.val = imagePrompts.val.filter(
                        (x) => x.x !== point.x && x.y !== point.y,
                      );
                      targetWidget.widgets.find(
                        (x) => x.name === 'image_prompts_json',
                      ).value = JSON.stringify(imagePrompts.val);
                      targetWidget.graph.change();
                    },
                  }),
                );
              }),
            );
          },
        ),
      );
    };

    const Container = () => {
      return div(
        {
          class: 'fixed left-0 top-0 w-full h-full z-[200] pointer-events-none',
          id: 'avatech-editor',
        },
        AvatarEditor(),
        ImageEditor(),
      );
    };

    van.add(document.body, Container());
  },

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === 'ExportGLTF') {
      addMenuHandler(nodeType, function (_, options) {
        const output = app.nodeOutputs[this.id + ''];
        if (!output || !output.gltfFilename) return;

        const gltfFilename =
          window.location.protocol +
          '//' +
          api.api_host +
          api.api_base +
          `/view?filename=${output.gltfFilename[0]}`;

        options.unshift({
          content: 'Save file',
          callback: () => {
            const a = document.createElement('a');
            let url = new URL(gltfFilename);
            url.searchParams.delete('preview');
            a.href = url;
            a.setAttribute(
              'download',
              new URLSearchParams(url.search).get('filename'),
            );
            document.body.append(a);
            a.click();
            requestAnimationFrame(() => a.remove());
          },
        });

        options.unshift({
          content: 'Open In Avatech Editor (Local)',
          callback: () => {
            openInAvatechEditor('http://localhost:3006', gltfFilename);
          },
        });

        options.unshift({
          content: 'Open In Avatech Editor',
          callback: () => {
            openInAvatechEditor('https://editor.avatech.ai', gltfFilename);
          },
        });
      });
    } else if (nodeData.name === 'SAM_Prompt_Image') {
      nodeData.input.required.upload = ['IMAGEUPLOAD'];
      // nodeData.input.required.prompts_points = ["IMAGEUPLOAD"];
      addExtraButton(nodeType, function () {

      });
      addMenuHandler(nodeType, function (_, options) {
        options.unshift({
          content: 'Open In Points Editor (Local)',
          callback: () => {
            showImageEditor.val = true;

            imagePrompts.val = JSON.parse(
              this.widgets.find((x) => x.name === 'image_prompts_json').value,
            );
            imageUrl.val = api.apiURL(
              `/view?filename=${encodeURIComponent(
                this.widgets.find((x) => x.name === 'image').value,
              )}&type=input&subfolder=`,
            );
            targetWidget = this;
            console.log(imageUrl.val, api.api_base);

            console.log(this);
          },
        });
      });
    }
  },
});
