import {
  iframeSrc,
  showEditor,
  showImageEditor,
  imageUrl,
  imagePrompts,
  targetNode,
} from './state.js';
import { van } from './van.js';
import { app } from './app.js';
import { api } from './api.js';
import { Container } from './Container.js';

/**
 * Adds a menu handler to a node type.
 * @param {Object} nodeType - The type of the node.
 * @param {Function} cb - The callback function to handle the menu.
 */
function addMenuHandler(nodeType, cb) {
  const getOpts = nodeType.prototype.getExtraMenuOptions;
  nodeType.prototype.getExtraMenuOptions = function () {
    const r = getOpts.apply(this, arguments);
    cb.apply(this, arguments);
    return r;
  };
}

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

/** @typedef {import('../../../web/types/comfy.js').ComfyExtension} ComfyExtension*/
/** @type {ComfyExtension} */
const ext = {
  getCustomWidgets(app) {
    return {
      SAM_PROMPTS(node, inputName, inputData, app) {
        const btn = node.addWidget('button', 'Edit prompt', '', () => {
          showMyImageEditor(node);
        });
        btn.serialize = false;

        return {
          widget: btn,
        };
      },
    };
  },
  
  name: 'Avatech.Avatar.BlendshapeEditor',

  init(app) {},

  async setup() {
    const graphCanvas = document.getElementById('graph-canvas');
    window.addEventListener(
      'keydown',
      (event) => {
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
      },
      {
        capture: true,
      },
    );

    window.addEventListener('keydown', (event) => {
      // if enter
      if (event.key === 'Enter') {
        event.preventDefault();
        document.getElementById('queue-button').click();
      }
    });

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
      nodeData.input.required.sam = ['SAM_PROMPTS'];
      nodeData.input.required.upload = ['IMAGEUPLOAD'];
      // nodeData.input.required.prompts_points = ["IMAGEUPLOAD"];
      addMenuHandler(nodeType, function (_, options) {
        options.unshift({
          content: 'Open In Points Editor (Local)',
          callback: () => {
            showMyImageEditor(this);
          },
        });
      });
    }
  },
};

function showMyImageEditor(node) {
  showImageEditor.val = true;

  imagePrompts.val = JSON.parse(
    node.widgets.find((x) => x.name === 'image_prompts_json').value,
  );
  imageUrl.val = api.apiURL(
    `/view?filename=${encodeURIComponent(
      node.widgets.find((x) => x.name === 'image').value,
    )}&type=input&subfolder=`,
  );
  targetNode.val = node;
  // console.log(imageUrl.val, api.api_base);
  // console.log(this);
}

app.registerExtension(ext);
