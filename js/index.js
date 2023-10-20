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
  showLoading,
  loadingCaption,
  alertDialog,
  showPreview,
  shareLoading,
  previewModelId,
} from "./state.js";
import { van } from "./van.js";
import { app } from "./app.js";
import { api } from "./api.js";
import { Container } from "./Container.js";
import { initModel, loadNpyTensor } from "./onnx.js";
import "https://code.iconify.design/3/3.1.0/iconify.min.js";
import { drawSegment, getClicks } from "./LayerEditor.js";
import { infoDialog } from "./dialog.js";

const stylesheet = document.createElement("link");
stylesheet.setAttribute("type", "text/css");
stylesheet.setAttribute("rel", "stylesheet");
stylesheet.setAttribute("href", "./avatar-graph-comfyui/tw-styles.css");
document.head.appendChild(stylesheet);

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
      blendshapes: targetNode.val.widgets.find((x) => x.name === "shape_flow")
        .value,
    },
    "*"
  );
}

function updateBlendshapesPrompts(value) {
  targetNode.val.widgets.find((x) => x.name === "shape_flow").value = value;
  targetNode.val.graph.change();
}

function getInputWidgetValue(node, inputIndex, widgetName) {
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
  let connectedImageFileName = getInputWidgetValue(node, 0, "image");
  if (!connectedImageFileName) {
    alertDialog.val = {
      text: "Please connect an image first",
      time: 3000,
    };
    return;
  }

  loadingCaption.val = "Loading SAM model...";
  showLoading.val = true;

  const ckpt = node.widgets.find((x) => x.name === "ckpt").value;
  const modelType = ckpt.match(/vit_[lbh]/)?.[0];
  initModel(modelType).then((res) => {
    loadingCaption.val = "Computing image embedding...";

    const split = connectedImageFileName.split("/");
    let id = connectedImageFileName;
    if (split.length > 1) id = split[1];

    node.widgets.find((x) => x.name === "embedding_id").value = id;

    api
      .fetchApi("/sam_model", {
        method: "POST",
        body: JSON.stringify({
          image: connectedImageFileName,
          embedding_id: id,
          ckpt,
        }),
      })
      .then(() => {
        showLoading.val = false;
        const v = JSON.parse(
          node.widgets.find((x) => x.name === "image_prompts_json").value
        );

        if (!Array.isArray(v)) {
          // this is a multi prompt
          imagePromptsMulti.val = v;
          selectedLayer.val = Object.keys(imagePromptsMulti.val)[0];
          imagePrompts.val = imagePromptsMulti.val[selectedLayer.val];
        } else {
          // this is a single prompt
          selectedLayer.val = "";
          imagePromptsMulti.val = {};
          imagePrompts.val = v;
        }
        showImageEditor.val = true;
        imageUrl.val = api.apiURL(
          `/view?filename=${encodeURIComponent(
            connectedImageFileName
          )}&type=input&subfolder=${split.length > 1 ? split[0] : ""}`
        );
        const embeedingUrl = api.apiURL(
          `/view?filename=${encodeURIComponent(
            `${id}_${modelType}.npy`
          )}&type=output&subfolder=`
        );
        loadNpyTensor(embeedingUrl).then((tensor) => {
          embeddings.val = tensor;
          drawSegment(getClicks());
        });
        targetNode.val = node;
      })
      .catch((err) => {
        console.log(err);
        showLoading.val = false;
      });
  });
}

/** @typedef {import('../../../web/types/comfy.js').ComfyExtension} ComfyExtension*/
/** @type {ComfyExtension} */
const ext = {
  getCustomWidgets(app) {
    return {
      SAM_PROMPTS(node, inputName, inputData, app) {
        const btn = node.addWidget("button", "Edit prompt", "", () => {
          showMyImageEditor(node);
          btn.serialize = false;
        });
        return {
          widget: btn,
        };
      },
      BLENDSHAPES_CONFIG(node, inputName, inputData, app) {
        const btn = node.addWidget("button", "Edit Shape Flow", "", () => {
          targetNode.val = node;
          openInAvatechEditor(
            "https://editor.avatech.ai?comfyui=true",
            fileName.val
          );
          // openInAvatechEditor("http://localhost:3006?comfyui=true", fileName.val);
        });
        btn.serialize = false;

        return {
          widget: btn,
        };
      },
      MESH_GROUP_CONFIG(node, inputName, inputData, app) {
        const btn = node.addWidget("button", "Add Mesh", "", () => {
          node.addInput("BPY_OBJ" + (node.inputs.length + 1), "BPY_OBJ");
          node.graph.change();
        });
        btn.serialize = false;

        return {
          widget: btn,
        };
      },
      MESH_GROUP_DELETE(node, inputName, inputData, app) {
        const btn = node.addWidget("button", "Delete Mesh", "", () => {
          console.log(node.inputs.length);
          node.removeInput(node.inputs.length - 1);
          node.graph.change();
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
    injectUIComponentToComfyuimenu();
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

        if (
          gltfFilename.endsWith(".ava") &&
          evt.detail?.output.auto_save[0] == "true"
        ) {
          const link = document.createElement("a");
          link.href = gltfFilename;
          link.download = gltfFilename.split("/").pop();
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        }

        fileName.val = gltfFilename;
        viewer.postMessage(
          JSON.stringify({
            avatarURL: gltfFilename,
            blendshapes: evt.detail?.output.SHAPE_FLOW[0],
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
          if (my_modal_3.open) {
            my_modal_3.close();
          } else {
            showImageEditor.val = false;
            showEditor.val = false;
          }
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
    switch (nodeData.name) {
      case "AvatarMainOutput":
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
        });
        break;
      case "ExportGLTF":
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
              openInAvatechEditor(
                "http://localhost:3006?comfyui=true",
                gltfFilename
              );
            },
          });

          options.unshift({
            content: "Open In Avatech Editor",
            callback: () => {
              openInAvatechEditor(
                "https://editor.avatech.ai?comfyui=true",
                gltfFilename
              );
            },
          });
        });
        break;
      case "SAM MultiLayer":
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
        break;
      case "CreateShapeFlow":
        nodeData.input.required.blendshape = ["BLENDSHAPES_CONFIG"];
        break;
      case "Mesh_JoinMesh":
        nodeData.input.required.obj = ["MESH_GROUP_CONFIG"];
        nodeData.input.required.del_obj = ["MESH_GROUP_DELETE"];
        break;
      default:
        break;
    }
  },
};

async function uploadPreview() {
  if (fileName.val == "")
    app.ui.dialog.show("Please create your avatar first.");
  else {
    const file = await fetch(fileName.val).then((e) => e.arrayBuffer());
    const model = await fetch("https://labs.avatech.ai/api/share", {
      method: "POST",
      body: file,
    }).then((e) => e.json());

    infoDialog.show(
      `Preview avatar url: <a href='https://editor.avatech.ai/viewer?objectId=${model.model_id}' target="_blank">https://editor.avatech.ai/viewer?objectId=` +
        model.model_id +
        `</a>`,
    );
    previewModelId.val = model.model_id;
  }
}

function injectUIComponentToComfyuimenu() {
  const menu = document.querySelector(".comfy-menu");
  const avatarPreview = document.createElement("button");
  avatarPreview.textContent = "Avatar Preview";
  avatarPreview.onclick = () => {
    showPreview.val = !showPreview.val;
  };

  const dropdown = document.createElement("div");
  dropdown.textContent = "▼";
  dropdown.className = "dropdownbtn";
  dropdown.onclick = (e) => {
    e.preventDefault();
    e.stopPropagation();

    LiteGraph.closeAllContextMenus();
    const menu = new LiteGraph.ContextMenu(
      [
        {
          title: "Create new share link",
          callback: async () => {
            shareAvatar.textContent = "Loading...";
            shareAvatar.append(dropdown);

            await uploadPreview();

            shareLoading.val = false;
            shareAvatar.textContent = "Share Avatar";
            shareAvatar.append(dropdown);
          },
        },
        {
          title: "Update avatar in current share link",
          callback: async () => {
            if (!previewModelId.val)
              app.ui.dialog.show("Please share your avatar first.");
            else {
              if (shareLoading.val) return;

              shareLoading.val = true;
              shareAvatar.textContent = "Loading...";
              shareAvatar.append(dropdown);

              const file = await fetch(fileName.val).then((e) =>
                e.arrayBuffer(),
              );
              const model = await fetch(
                "https://labs.avatech.ai/api/share?id=" + previewModelId.val,
                {
                  method: "POST",
                  body: file,
                },
              ).then((e) => e.json());

              infoDialog.show(
                `Preview updated: <a href='https://editor.avatech.ai/viewer?objectId=${model.model_id}' target="_blank">https://editor.avatech.ai/viewer?objectId=` +
                  model.model_id +
                  "</a>\n Remember to hard refresh before checking out the new preview!",
              );

              shareLoading.val = false;
              shareAvatar.textContent = "Share Avatar";
              shareAvatar.append(dropdown);
            }
          },
        },
      ],

      {
        event: e,
        scale: 1.3,
      },
      window,
    );
    menu.root.classList.add("popup");
  };

  const shareAvatar = document.createElement("button");
  shareAvatar.textContent = "Share Avatar";
  shareAvatar.className = "sharebtn";
  shareAvatar.onclick = async () => {
    if (shareLoading.val) return;

    if (!previewModelId.val) {
      shareLoading.val = true;
      shareAvatar.textContent = "Loading...";
      shareAvatar.append(dropdown);

      await uploadPreview();

      shareLoading.val = false;
      shareAvatar.textContent = "Share Avatar";
      shareAvatar.append(dropdown);
    } else {
      infoDialog.show(
        `Preview avatar url: <a href='https://editor.avatech.ai/viewer?objectId=${previewModelId.val}' target="_blank">https://editor.avatech.ai/viewer?objectId=` +
          previewModelId.val +
          `</a>`,
      );
    }
  };

  menu.append(avatarPreview);
  menu.append(shareAvatar);

  shareAvatar.append(dropdown);
}

app.registerExtension(ext);
