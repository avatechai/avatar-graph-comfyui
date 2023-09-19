import {
  iframeSrc,
  showEditor,
  showImageEditor,
  imageUrl,
  imagePrompts,
  targetNode,
  fileName,
  embeddings,
  imagePromptsMulti,
  selectedLayer,
} from "./state.js";
import { van } from "./van.js";
import { app } from "./app.js";
import { api } from "./api.js";
import { Container } from "./Container.js";
import { loadNpyTensor } from "./onnx.js";
import "https://code.iconify.design/3/3.1.0/iconify.min.js";
import { drawSegment, getClicks } from "./ImageEditor.js";

/** @type {import( '../../../web/types/litegraph.js').LGraphGroup} */
const recomputeInsideNodesOps = LGraphGroup.prototype.recomputeInsideNodes;
LGraphGroup.prototype.recomputeInsideNodes = function () {
  this._nodes.length = 0;
  var nodes = this.graph._nodes;
  var node_bounding = new Float32Array(4);

  // const r = recomputeInsideNodesOps.apply(this, arguments);
  for (var i = 0; i < nodes.length; ++i) {
    var node = nodes[i];
    node.getBounding(node_bounding);
    if (!LiteGraph.overlapBounding(this._bounding, node_bounding)) {
      if (node.parentId != undefined && node.parentId == this.id) {
        node.parentId = null;
      }
      continue;
    } //out of the visible area
    this._nodes.push(node);
  }

  this.repositionNodes();
};

const nodeSer = LGraphNode.prototype.serialize;
LGraphNode.prototype.serialize = function () {
  const r = nodeSer.apply(this, arguments);
  r.parentId = this.parentId;
  return r;
};

LGraphGroup.prototype.repositionNodes = function () {
  if (!this.isStack) return;

  const pos = [this.pos[0] + 10, this.pos[1] + 80];

  let height = 0;
  let width = this.size[0];

  let sortedNodes = this._nodes.sort((a, b) => a.pos[1] - b.pos[1]);
  // Separate input and output nodes from the rest
  const inputNodes = sortedNodes.filter(
    (node) => node.properties.routeType === "input"
  );
  const outputNodes = sortedNodes.filter(
    (node) => node.properties.routeType === "output"
  );
  const otherNodes = sortedNodes.filter(
    (node) =>
      node.properties.routeType !== "input" &&
      node.properties.routeType !== "output"
  );

  // Concatenate the arrays so that input nodes are first and output nodes are last
  sortedNodes = [...inputNodes, ...otherNodes, ...outputNodes];

  for (var i = 0; i < sortedNodes.length; ++i) {
    /** @type {LGraphNode} */
    var node = sortedNodes[i];
    node.pos[0] = pos[0];
    node.pos[1] = pos[1] + height;

    node.parentId = this.id;

    if (node.type !== "Reroute") node.size[0] = width - 20;

    if (node.flags.collapsed) {
      height += 40;
    } else if (node.type === "Reroute") {
      height += 30;
      node.pos[1] -= 30;
    } else {
      height += 40;
      height += node.size[1];
    }

    // connect this node output
    if (i < sortedNodes.length - 1) {
      node.disconnectOutput(0);
      node.connect(0, sortedNodes[i + 1], 0);
    }
  }

  this.size[1] = height + 60;
};

const collapseOps = LGraphNode.prototype.collapse;
LGraphNode.prototype.collapse = function (force) {
  collapseOps.apply(this, arguments);
  this.computeParentGroupResize();
};

LGraphNode.prototype.computeParentGroupResize = function () {
  if (this.parentId) {
    const parent = this.graph._groups.find((x) => x.id === this.parentId);
    if (parent) {
      parent.recomputeInsideNodes();
    }
  }
};

const getOpts = LGraphCanvas.prototype.getCanvasMenuOptions;
LGraphCanvas.prototype.getCanvasMenuOptions = function () {
  const r = getOpts.apply(this, arguments);
  r.push({
    content: "Add Stack",
    callback: (info, entry, mouse_event) => {
      var canvas = LGraphCanvas.active_canvas;
      var ref_window = canvas.getCanvasWindow();

      var group = new LiteGraph.LGraphGroup();
      group.pos = canvas.convertEventToCanvasOffset(mouse_event);
      group.isStack = true;
      group.title = "Stack";
      canvas.graph.add(group);

      // add two reroute nodes
      var reroute1 = LiteGraph.createNode("Reroute");
      var reroute2 = LiteGraph.createNode("Reroute");
      reroute1.properties.routeType = "input";
      reroute2.properties.routeType = "output";
      reroute1.pos = [group.pos[0] + 10, group.pos[1] + 40];
      reroute2.pos = [group.pos[0] + 10, group.pos[1] + 70];
      canvas.graph.add(reroute1);
      canvas.graph.add(reroute2);
    },
  });
  return r;
};

const ctor = LGraphGroup.prototype._ctor;
LGraphGroup.prototype._ctor = function (title) {
  ctor.apply(this, arguments);
  this.isStack = false;
  this.id = LiteGraph.uuidv4();
};

const serializationOps = LGraphGroup.prototype.serialize;
LGraphGroup.prototype.serialize = function () {
  const r = serializationOps.apply(this, arguments);
  r.id = this.id;
  r.isStack = this.isStack;
  return r;
};

const configureOps = LGraphGroup.prototype.configure;
LGraphGroup.prototype.configure = function (o) {
  configureOps.apply(this, arguments);
  this.id = o.id;
  this.isStack = o.isStack;
};

/**
 * @typedef {import('../../../web/types/litegraph.js').LGraph} LGraph
 * @typedef {import('../../../web/types/litegraph.js').LGraphNode} LGraphNode
 */

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
  let editor = document.getElementById("avatech-editor-iframe");
  iframeSrc.val = url;
  showEditor.val = true;

  editor.contentWindow.postMessage(
    {
      key: "key",
      value: fileName,
      method: "store",
    },
    "*"
  );
}

function updateBlendshapesPrompts(value) {
  targetNode.val.widgets.find((x) => x.name === "blendshapes").value = value;
  targetNode.val.graph.change();
}

function getWidgetValue(node, inputIndex, widgetName) {
  /** @type {LGraph} */
  const graph = app.graph;

  const nodeLink = node.inputs[inputIndex].link;
  if (!nodeLink) return;

  const targetLink = graph.links[nodeLink];

  /** @type {LGraphNode} */
  let nodea = graph._nodes_by_id[targetLink.origin_id];

  while (nodea.type == "Reroute") {
    nodea = nodea.getInputNode(0);
  }

  console.log(targetLink, nodea);
  console.log(nodea.getInputNode(0, true));

  /** @type {string} */
  return nodea.widgets.find((x) => x.name === widgetName).value;
}

/**
 *
 * @param {LGraphNode} node
 */
function showMyImageEditor(node) {
  let connectedImageFileName = getWidgetValue(node, 0, "image");
  const split = connectedImageFileName.split("/");
  if (split.length > 1) connectedImageFileName = split[1];

  const connectedEmbeddingFileName = getWidgetValue(node, 1, "embedding_id");

  showImageEditor.val = true;

  imagePrompts.val = JSON.parse(
    node.widgets.find((x) => x.name === "image_prompts_json").value
  );
  if (!Array.isArray(imagePrompts.val)) {
    imagePromptsMulti.val = imagePrompts.val;

    selectedLayer.val = Object.keys(imagePromptsMulti.val)[0];

    imagePrompts.val = imagePromptsMulti.val[selectedLayer.val];
  } else {
    imagePromptsMulti.val = {};
  }
  imageUrl.val = api.apiURL(
    `/view?filename=${encodeURIComponent(
      connectedImageFileName
    )}&type=input&subfolder=${split.length > 1 ? split[0] : ""}`
  );
  const embeedingUrl = api.apiURL(
    `/view?filename=${encodeURIComponent(
      `${connectedEmbeddingFileName}.npy`
    )}&type=output&subfolder=`
  );
  loadNpyTensor(embeedingUrl).then((tensor) => {
    embeddings.val = tensor;
    drawSegment(getClicks());
  });
  targetNode.val = node;
}

/** @typedef {import('../../../web/types/comfy.js').ComfyExtension} ComfyExtension*/
/** @type {ComfyExtension} */
const ext = {
  getCustomWidgets(app) {
    return {
      SAM_PROMPTS(node, inputName, inputData, app) {
        const btn = node.addWidget("button", "Edit prompt", "", () => {
          showMyImageEditor(node);
        });
        btn.serialize = false;

        return {
          widget: btn,
        };
      },
      BLENDSHAPES_CONFIG(node, inputName, inputData, app) {
        const btn = node.addWidget("button", "Edit blendshapes", "", () => {
          targetNode.val = node;
          openInAvatechEditor("https://editor.avatech.ai", fileName.val);
          // openInAvatechEditor("http://localhost:3006", fileName.val);
        });
        btn.serialize = false;

        return {
          widget: btn,
        };
      },
    };
  },

  name: "Avatech.Avatar.BlendshapeEditor",

  init(app) {
    const onNodeMoved = app.canvas.onNodeMoved;
    app.canvas.onNodeMoved = function (node) {
      const r = onNodeMoved?.apply(this, arguments);

      app.graph._groups.forEach((x) => {
        x.recomputeInsideNodes();
      });

      node.computeParentGroupResize();
    };
  },

  async setup() {
    const graphCanvas = document.getElementById("graph-canvas");

    window.addEventListener("message", (event) => {
      if (!event.data.flow || Object.entries(event.data.flow).length <= 0)
        return;
      updateBlendshapesPrompts(event.data.flow);
    });

    api.addEventListener("executed", (evt) => {
      if (evt.detail?.output.gltfFilename) {
        const viewer = document.getElementById(
          "avatech-viewer-iframe"
        ).contentWindow;

        const gltfFilename =
          window.location.protocol +
          "//" +
          api.api_host +
          api.api_base +
          `/view?filename=${evt.detail?.output.gltfFilename[0]}`;

        fileName.val = gltfFilename;
        viewer.postMessage(
          JSON.stringify({
            avatarURL: gltfFilename,
            blendshapes: evt.detail?.output.blendshapes[0],
          }),
          "*"
        );
      }
    });

    window.addEventListener(
      "keydown",
      (event) => {
        if (event.key === "Escape") {
          event.preventDefault();
          showImageEditor.val = false;
        }
      },
      {
        capture: true,
      }
    );

    graphCanvas.addEventListener("keydown", (event) => {
      if (event.key === "b") {
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

      if (event.key === "v" && !event.ctrlKey && !event.metaKey) {
        event.preventDefault();
        const currentGraph = app.graph.list_of_graphcanvas[0];
        if (currentGraph.selected_nodes.length !== 1) {
          Object.values(currentGraph.selected_nodes).forEach((targetNode) => {
            if (targetNode.flags.collapsed) targetNode.flags.collapsed = false;
            else targetNode.flags.collapsed = true;

            targetNode.computeParentGroupResize();
          });
        } else {
          const targetNode = currentGraph.current_node;
          console.log(currentGraph.selected_nodes);
          if (targetNode.flags.collapsed) targetNode.flags.collapsed = false;
          else targetNode.flags.collapsed = true;

          targetNode.computeParentGroupResize();
        }
        app.graph.change();
      }
    });

    graphCanvas.addEventListener("keydown", (event) => {
      // if enter
      if (event.key === "Enter") {
        event.preventDefault();
        document.getElementById("queue-button").click();
      }
    });

    van.add(document.body, Container());
  },

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "ExportGLTF") {
      addMenuHandler(nodeType, function (_, options) {
        const output = app.nodeOutputs[this.id + ""];
        if (!output || !output.gltfFilename) return;

        const gltfFilename =
          window.location.protocol +
          "//" +
          api.api_host +
          api.api_base +
          `/view?filename=${output.gltfFilename[0]}`;

        options.unshift({
          content: "Save file",
          callback: () => {
            const a = document.createElement("a");
            let url = new URL(gltfFilename);
            url.searchParams.delete("preview");
            a.href = url;
            a.setAttribute(
              "download",
              new URLSearchParams(url.search).get("filename")
            );
            document.body.append(a);
            a.click();
            requestAnimationFrame(() => a.remove());
          },
        });

        options.unshift({
          content: "Open In Avatech Editor (Local)",
          callback: () => {
            openInAvatechEditor("http://localhost:3006", gltfFilename);
          },
        });

        options.unshift({
          content: "Open In Avatech Editor",
          callback: () => {
            openInAvatechEditor("https://editor.avatech.ai", gltfFilename);
          },
        });
      });
    } else if (nodeData.name === "SAM_Prompt_Image") {
      nodeData.input.required.sam = ["SAM_PROMPTS"];
      // nodeData.input.required.upload = ['IMAGEUPLOAD'];
      // nodeData.input.required.prompts_points = ["IMAGEUPLOAD"];
      addMenuHandler(nodeType, function (_, options) {
        options.unshift({
          content: "Open In Points Editor (Local)",
          callback: () => {
            showMyImageEditor(this);
          },
        });
      });
    } else if (nodeData.name === "ExportBlendshapes") {
      nodeData.input.required.blendshape = ["BLENDSHAPES_CONFIG"];
      addMenuHandler(nodeType, function (_, options) {
        options.unshift({
          content: "Open In Blendshapes Editor",
          callback: () => {
            openInAvatechEditor("https://editor.avatech.ai", gltfFilename);
          },
        });
      });
    }
  },
};

app.registerExtension(ext);
