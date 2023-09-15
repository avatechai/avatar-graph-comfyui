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

  const sortedNodes = this._nodes.sort((a, b) => a.pos[1] - b.pos[1]);

  for (var i = 0; i < sortedNodes.length; ++i) {
    /** @type {LGraphNode} */
    var node = sortedNodes[i];
    node.pos[0] = pos[0];
    node.pos[1] = pos[1] + height;

    node.parentId = this.id;
    node.size[0] = this.size[0] - 20;

    height += 40;

    if (node.flags.collapsed) {
    } else {
      height += node.size[1];
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
    content: 'Add Stack',
    callback: (info, entry, mouse_event) => {
      var canvas = LGraphCanvas.active_canvas;
      var ref_window = canvas.getCanvasWindow();

      var group = new LiteGraph.LGraphGroup();
      group.pos = canvas.convertEventToCanvasOffset(mouse_event);
      group.isStack = true;
      group.title = 'Stack';
      canvas.graph.add(group);
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

/**
 *
 * @param {LGraphNode} node
 */
function showMyImageEditor(node) {
  /** @type {LGraph} */
  const graph = app.graph;

  const imageNodeLink = node.inputs[0].link;
  if (!imageNodeLink) return;

  const targetLink = graph.links[imageNodeLink];

  /** @type {LGraphNode} */
  let nodea = graph._nodes_by_id[targetLink.origin_id];

  while (nodea.type == 'Reroute') {
    nodea = nodea.getInputNode(0);
  }

  console.log(targetLink, nodea);
  console.log(nodea.getInputNode(0, true));

  /** @type {string} */
  let connectedImageFileName = nodea.widgets.find(
    (x) => x.name === 'image',
  ).value;

  const split = connectedImageFileName.split('/');

  if (split.length > 1) connectedImageFileName = split[1];

  console.log(connectedImageFileName);

  // node.widgets.find((x) => x.name === 'image').value

  showImageEditor.val = true;

  imagePrompts.val = JSON.parse(
    node.widgets.find((x) => x.name === 'image_prompts_json').value,
  );
  imageUrl.val = api.apiURL(
    `/view?filename=${encodeURIComponent(
      connectedImageFileName,
    )}&type=input&subfolder=${split.length > 1 ? split[0] : ''}`,
  );
  targetNode.val = node;
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

  init(app) {
    const onNodeMoved = app.canvas.onNodeMoved;
    app.canvas.onNodeMoved = function (node) {
      const r = onNodeMoved?.apply(this, arguments);

      node.computeParentGroupResize();
    };
  },

  async setup() {
    const graphCanvas = document.getElementById('graph-canvas');

    api.addEventListener('executed', (evt) => {
      if (evt.detail?.output.gltfFilename) {
        const viewer = document.getElementById(
          'avatech-viewer-iframe'
        ).contentWindow

        const gltfFilename =
          window.location.protocol +
          '//' +
          api.api_host +
          api.api_base +
          `/view?filename=${evt.detail?.output.gltfFilename[0]}`;


        viewer.postMessage(
          JSON.stringify({
            avatarURL: gltfFilename,
            blendshapes:
              '{"nodes":[{"width":160,"height":93,"id":"modelBlendshapesOutput_fnNms","data":{"label":"modelBlendshapesOutput_fnNms","variableName":"modelBlendshapesOutput_fnNms","defaultValues":{"model_blendshapes_Ex_6":{"type":"number","value":0},"model_blendshapes_Ex_Q":{"type":"number","value":0},"model_blendshapes_Ex_i":{"type":"number","value":0},"model_blendshapes_BrowY":{"type":"number","value":0},"model_blendshapes_EX_QQ":{"type":"number","value":0},"model_blendshapes_HeadX":{"type":"number","value":0},"model_blendshapes_HeadY":{"type":"number","value":0},"model_blendshapes_Param":{"type":"number","value":0},"model_blendshapes_Breath":{"type":"number","value":1.8282266430674599},"model_blendshapes_EX_zzz":{"type":"number","value":0},"model_blendshapes_MouthX":{"type":"number","value":0},"model_blendshapes_MouthY":{"type":"number","value":0},"model_blendshapes_Ex_Cute":{"type":"number","value":0},"model_blendshapes_Ex_Tear":{"type":"number","value":0},"model_blendshapes_Ex_drop":{"type":"number","value":0},"model_blendshapes_EyeWide":{"type":"number","value":0},"model_blendshapes_JawOpen":{"type":"number","value":0},"model_blendshapes_EX_Blush":{"type":"number","value":0},"model_blendshapes_EX_Point":{"type":"number","value":0},"model_blendshapes_EX_Sweat":{"type":"number","value":0},"model_blendshapes_EyeLookX":{"type":"number","value":0},"model_blendshapes_EyeLookY":{"type":"number","value":0},"model_blendshapes_BrowInner":{"type":"number","value":0},"model_blendshapes_BrowOuter":{"type":"number","value":0},"model_blendshapes_CheekPuff":{"type":"number","value":0},"model_blendshapes_EX_Shadow":{"type":"number","value":0},"model_blendshapes_EyeBlinkL":{"type":"number","value":0},"model_blendshapes_EyeBlinkR":{"type":"number","value":0},"model_blendshapes_EyeSquint":{"type":"number","value":0},"model_blendshapes_FX_Hair01":{"type":"number","value":0},"model_blendshapes_FX_Hair02":{"type":"number","value":0},"model_blendshapes_FX_Hair03":{"type":"number","value":0},"model_blendshapes_MouthForm":{"type":"number","value":0},"model_blendshapes_MouthWide":{"type":"number","value":0},"model_blendshapes_NoseSneer":{"type":"number","value":0},"model_blendshapes_tongueOut":{"type":"number","value":0},"model_blendshapes_EX_FireEye":{"type":"number","value":0},"model_blendshapes_EX_StarEye":{"type":"number","value":0},"model_blendshapes_EX_TearEye":{"type":"number","value":0},"model_blendshapes_MouthClose":{"type":"number","value":1},"model_blendshapes_CheekSquint":{"type":"number","value":0},"model_blendshapes_EX_DizzyEye":{"type":"number","value":0},"model_blendshapes_EX_HeartEye":{"type":"number","value":0},"model_blendshapes_Ex_wideopen":{"type":"number","value":0},"model_blendshapes_EyeL_Squint":{"type":"number","value":0},"model_blendshapes_EyeR_Squint":{"type":"number","value":0},"model_blendshapes_MouthPucker":{"type":"number","value":0},"model_blendshapes_ParamAngleX":{"type":"number","value":0},"model_blendshapes_ParamAngleY":{"type":"number","value":0},"model_blendshapes_ParamAngleZ":{"type":"number","value":0},"model_blendshapes_ParamBrowLY":{"type":"number","value":0},"model_blendshapes_ParamBrowRY":{"type":"number","value":0},"model_blendshapes_ParamMouthX":{"type":"number","value":0},"model_blendshapes_FX_EyeblinkL":{"type":"number","value":0},"model_blendshapes_FX_EyeblinkR":{"type":"number","value":0},"model_blendshapes_ParamJawOpen":{"type":"number","value":0},"model_blendshapes_EX_FaceShadow":{"type":"number","value":0},"model_blendshapes_ParamEyeBallX":{"type":"number","value":0},"model_blendshapes_ParamEyeBallY":{"type":"number","value":0},"model_blendshapes_ParamEyeLClose":{"type":"number","value":0},"model_blendshapes_ParamEyeRClose":{"type":"number","value":0},"model_blendshapes_ParamMouthForm":{"type":"number","value":0},"model_blendshapes_ParamMouthOpenY":{"type":"number","value":0},"model_blendshapes_ParamMouthShrug":{"type":"number","value":0},"model_blendshapes_ParamMouthFunnel":{"type":"number","value":0},"model_blendshapes_ParamMouthPuckerWiden":{"type":"number","value":0},"model_blendshapes_ParamMouthPressLipOpen":{"type":"number","value":0},"model_blendshapes_EyeBlinkLeft":{"value":0,"type":"number"},"model_blendshapes_EyeBlinkRight":{"value":0,"type":"number"}}},"type":"modelBlendshapesOutput","dragging":false,"position":{"x":1430.761299009229,"y":-615.5844816915322},"selected":false,"positionAbsolute":{"x":1430.761299009229,"y":-615.5844816915322}},{"width":166,"height":94,"id":"timeNode_jYYGz","data":{"time":0,"type":"currentTime","label":"timeNode_jYYGz","deltaTime":0.03110000000148716,"currentTime":51.64380000000447,"variableName":"timeNode_jYYGz"},"type":"timeNode","dragging":false,"position":{"x":-1446.5497048729528,"y":-1574.0075996592495},"selected":false,"positionAbsolute":{"x":-1446.5497048729528,"y":-1574.0075996592495}},{"width":172,"height":74,"id":"easingNode_bqAhh","data":{"label":"easingNode_bqAhh","easingFn":"easeInSine","variableName":"easingNode_bqAhh"},"type":"easingNode","dragging":false,"position":{"x":-393.38986226682505,"y":-1477.1168170532294},"selected":false,"positionAbsolute":{"x":-393.38986226682505,"y":-1477.1168170532294}},{"width":160,"height":147,"id":"mathNode_UuJyg","data":{"label":"mathNode_UuJyg","fnName":"min","variableName":"mathNode_UuJyg","defaultValues":{"math_fn_input_0":{"type":"number","value":0.9999999999999999},"math_fn_input_1":{"type":"number","value":1}}},"type":"mathNode","dragging":false,"position":{"x":-157.89744551276848,"y":-1394.2426827811544},"selected":false,"positionAbsolute":{"x":-157.89744551276848,"y":-1394.2426827811544}},{"width":160,"height":66,"id":"valueNode_rpCCe","data":{"max":1,"label":"valueNode_rpCCe","value":1,"variableName":"valueNode_rpCCe"},"type":"valueNode","dragging":false,"position":{"x":-389.881619144479,"y":-1222.839718448476},"selected":false,"positionAbsolute":{"x":-389.881619144479,"y":-1222.839718448476}},{"width":160,"height":147,"id":"operatorNode_nKzQk","data":{"label":"operatorNode_nKzQk","operator":2,"variableName":"operatorNode_nKzQk","defaultValues":{"operand_1":{"type":"number","value":122.37870000004769},"operand_2":{"type":"number","value":4.6}}},"type":"operatorNode","dragging":false,"position":{"x":-1467.302645495367,"y":-1330.5348641229261},"selected":false,"positionAbsolute":{"x":-1467.302645495367,"y":-1330.5348641229261}},{"width":160,"height":66,"id":"valueNode_Biisy","data":{"max":100,"label":"valueNode_Biisy","value":4.6,"variableName":"valueNode_Biisy"},"type":"valueNode","dragging":false,"position":{"x":-1783.939700117743,"y":-1238.9025531115467},"selected":false,"positionAbsolute":{"x":-1783.939700117743,"y":-1238.9025531115467}},{"width":160,"height":66,"id":"valueNode_gSXlE","data":{"max":10,"label":"valueNode_gSXlE","value":4.1,"variableName":"valueNode_gSXlE"},"type":"valueNode","dragging":false,"position":{"x":-1454.8054155891307,"y":-1756.2667855624913},"selected":false,"positionAbsolute":{"x":-1454.8054155891307,"y":-1756.2667855624913}},{"width":160,"height":147,"id":"operatorNode_hddlu","data":{"label":"operatorNode_hddlu","operator":2,"variableName":"operatorNode_hddlu","defaultValues":{"operand_1":{"type":"number","value":4.1},"operand_2":{"type":"number","value":122.37870000004769}}},"type":"operatorNode","dragging":false,"position":{"x":-1203.2636707550314,"y":-1634.9674546671938},"selected":false,"positionAbsolute":{"x":-1203.2636707550314,"y":-1634.9674546671938}},{"width":160,"height":66,"id":"valueNode_lYzmY","data":{"label":"valueNode_lYzmY","value":1,"variableName":"valueNode_lYzmY"},"type":"valueNode","dragging":false,"position":{"x":-192.2639928132918,"y":-1569.0701190207008},"selected":false,"positionAbsolute":{"x":-192.2639928132918,"y":-1569.0701190207008}},{"width":160,"height":147,"id":"operatorNode_neoYv","data":{"label":"operatorNode_neoYv","operator":1,"variableName":"operatorNode_neoYv","defaultValues":{"operand_1":{"type":"number","value":1},"operand_2":{"type":"number","value":0.9999999999999999}}},"type":"operatorNode","dragging":false,"position":{"x":93.35187855750155,"y":-1450.4246063120327},"selected":false,"positionAbsolute":{"x":93.35187855750155,"y":-1450.4246063120327}},{"width":160,"height":66,"id":"valueNode_UbOMo","data":{"label":"valueNode_UbOMo","value":1,"variableName":"valueNode_UbOMo"},"type":"valueNode","dragging":false,"position":{"x":-1180.8732646806407,"y":-955.1287853807431},"selected":false,"positionAbsolute":{"x":-1180.8732646806407,"y":-955.1287853807431}},{"width":166,"height":94,"id":"timeNode_cLYze","data":{"time":0,"type":"currentTime","label":"timeNode_cLYze","deltaTime":0.03110000000148716,"currentTime":51.64380000000447,"variableName":"timeNode_cLYze"},"type":"timeNode","dragging":false,"position":{"x":-1726.9796189626877,"y":-1416.2763563116484},"selected":false,"positionAbsolute":{"x":-1726.9796189626877,"y":-1416.2763563116484}},{"width":172,"height":74,"id":"easingNode_nGvYS","data":{"label":"easingNode_nGvYS","easingFn":"easeInSine","variableName":"easingNode_nGvYS"},"type":"easingNode","dragging":false,"position":{"x":-846.764780985111,"y":-1249.9808740621884},"selected":false,"positionAbsolute":{"x":-846.764780985111,"y":-1249.9808740621884}},{"width":160,"height":147,"id":"operatorNode_edAkt","data":{"label":"operatorNode_edAkt","operator":4,"variableName":"operatorNode_edAkt","defaultValues":{"operand_1":{"type":"number","value":501.75267000019545},"operand_2":{"type":"number","value":10}}},"type":"operatorNode","dragging":false,"position":{"x":-951.6077899817305,"y":-1514.623300323864},"selected":false,"positionAbsolute":{"x":-951.6077899817305,"y":-1514.623300323864}},{"width":160,"height":66,"id":"valueNode_ARGrY","data":{"max":100,"label":"valueNode_ARGrY","value":10,"variableName":"valueNode_ARGrY"},"type":"valueNode","dragging":false,"position":{"x":-1224.8896769064443,"y":-1416.9331452004221},"selected":false,"positionAbsolute":{"x":-1224.8896769064443,"y":-1416.9331452004221}},{"width":160,"height":200,"id":"conditionNode_NuykU","data":{"label":"conditionNode_NuykU","operator":0,"variableName":"conditionNode_NuykU","defaultValues":{"condition_a":{"type":"number","value":1.752670000195451},"condition_b":{"type":"number","value":1},"condition_true":{"type":"number","value":1.752670000195451},"condition_false":{"type":"number","value":1}}},"type":"conditionNode","dragging":false,"position":{"x":-674.6418145197106,"y":-1460.443595680259},"selected":false,"positionAbsolute":{"x":-674.6418145197106,"y":-1460.443595680259}},{"width":160,"height":66,"id":"valueNode_cPQRA","data":{"label":"valueNode_cPQRA","value":1,"variableName":"valueNode_cPQRA"},"type":"valueNode","dragging":false,"position":{"x":-976.9936401356814,"y":-1339.2858259698112},"selected":false,"positionAbsolute":{"x":-976.9936401356814,"y":-1339.2858259698112}},{"width":172,"height":74,"id":"easingNode_AIbfm","data":{"label":"easingNode_AIbfm","easingFn":"easeInOutSine","variableName":"easingNode_AIbfm"},"type":"easingNode","dragging":false,"position":{"x":301.5579330370372,"y":-1366.0821040000003},"selected":false,"positionAbsolute":{"x":301.5579330370372,"y":-1366.0821040000003}},{"width":166,"height":94,"id":"timeNode_vRvny","data":{"time":0,"type":"currentTime","label":"timeNode_vRvny","deltaTime":0.03110000000148716,"currentTime":51.64380000000447,"variableName":"timeNode_vRvny"},"type":"timeNode","dragging":false,"position":{"x":-689.8807442962964,"y":-811.3355333333344},"selected":false,"positionAbsolute":{"x":-689.8807442962964,"y":-811.3355333333344}},{"width":160,"height":200,"id":"conditionNode_gNWNm","data":{"label":"conditionNode_gNWNm","operator":0,"variableName":"conditionNode_gNWNm","defaultValues":{"condition_a":{"type":"number","value":3.478700000047697},"condition_b":{"type":"number","value":1},"condition_true":{"type":"number","value":1.1102230246251565e-16},"condition_false":{"type":"number","value":0}}},"type":"conditionNode","dragging":false,"position":{"x":545.2539044134492,"y":-1206.0414364381195},"selected":false,"positionAbsolute":{"x":545.2539044134492,"y":-1206.0414364381195}},{"width":160,"height":147,"id":"operatorNode_dNObC","data":{"label":"operatorNode_dNObC","operator":4,"variableName":"operatorNode_dNObC","defaultValues":{"operand_1":{"type":"number","value":122.37870000004769},"operand_2":{"type":"number","value":4.1}}},"type":"operatorNode","dragging":false,"position":{"x":-135.76963318518528,"y":-572.14108888889},"selected":false,"positionAbsolute":{"x":-135.76963318518528,"y":-572.14108888889}},{"width":160,"height":66,"id":"valueNode_ehGvn","data":{"max":10,"label":"valueNode_ehGvn","value":4.1,"variableName":"valueNode_ehGvn"},"type":"valueNode","dragging":false,"position":{"x":-414.9362998518519,"y":-552.6966444444455},"selected":false,"positionAbsolute":{"x":-414.9362998518519,"y":-552.6966444444455}},{"width":160,"height":147,"id":"operatorNode_uiGHN","data":{"label":"operatorNode_uiGHN","operator":2,"variableName":"operatorNode_uiGHN","defaultValues":{"operand_1":{"type":"number","value":122.37870000004769},"operand_2":{"type":"number","value":1}}},"type":"operatorNode","dragging":false,"position":{"x":-427.436299851852,"y":-722.9744222222234},"selected":false,"positionAbsolute":{"x":-427.436299851852,"y":-722.9744222222234}},{"width":160,"height":66,"id":"valueNode_ZlTlm","data":{"max":10,"label":"valueNode_ZlTlm","value":1,"variableName":"valueNode_ZlTlm"},"type":"valueNode","dragging":false,"position":{"x":-660.7696331851854,"y":-629.9188666666679},"selected":false,"positionAbsolute":{"x":-660.7696331851854,"y":-629.9188666666679}},{"width":160,"height":147,"id":"mathNode_TdXuG","data":{"label":"mathNode_TdXuG","fnName":"min","variableName":"mathNode_TdXuG","defaultValues":{"math_fn_input_0":{"type":"number","value":0.9999999999999999},"math_fn_input_1":{"type":"number","value":1}}},"type":"mathNode","dragging":false,"position":{"x":-653.0003887407406,"y":-1152.3043066666676},"selected":false,"positionAbsolute":{"x":-653.0003887407406,"y":-1152.3043066666676}},{"width":160,"height":66,"id":"valueNode_TpLsh","data":{"label":"valueNode_TpLsh","value":1,"variableName":"valueNode_TpLsh"},"type":"valueNode","dragging":false,"position":{"x":-1531.1029665185179,"y":-1026.6966444444454},"selected":false,"positionAbsolute":{"x":-1531.1029665185179,"y":-1026.6966444444454}},{"width":160,"height":147,"id":"operatorNode_rEaHg","data":{"label":"operatorNode_rEaHg","operator":4,"variableName":"operatorNode_rEaHg","defaultValues":{"operand_1":{"type":"number","value":562.9420200002193},"operand_2":{"type":"number","value":10}}},"type":"operatorNode","dragging":false,"position":{"x":-1268.491855407406,"y":-1244.4188666666676},"selected":false,"positionAbsolute":{"x":-1268.491855407406,"y":-1244.4188666666676}},{"width":160,"height":66,"id":"valueNode_FSQaP","data":{"max":10,"label":"valueNode_FSQaP","value":10,"variableName":"valueNode_FSQaP"},"type":"valueNode","dragging":false,"position":{"x":-1512.491855407406,"y":-1164.4188666666676},"selected":false,"positionAbsolute":{"x":-1512.491855407406,"y":-1164.4188666666676}},{"width":160,"height":200,"id":"conditionNode_pHuSn","data":{"label":"conditionNode_pHuSn","operator":0,"variableName":"conditionNode_pHuSn","defaultValues":{"condition_a":{"type":"number","value":2.9420200002192587},"condition_b":{"type":"number","value":1},"condition_true":{"type":"number","value":2.9420200002192587},"condition_false":{"type":"number","value":1}}},"type":"conditionNode","dragging":false,"position":{"x":-1030.491855407406,"y":-1184.3347706666675},"selected":false,"positionAbsolute":{"x":-1030.491855407406,"y":-1184.3347706666675}},{"width":160,"height":147,"id":"operatorNode_oeTEr","data":{"label":"operatorNode_oeTEr","operator":1,"variableName":"operatorNode_oeTEr","defaultValues":{"operand_1":{"type":"number","value":1},"operand_2":{"type":"number","value":0.9999999999999999}}},"type":"operatorNode","dragging":false,"position":{"x":-468.2721754074064,"y":-1095.4132186666675},"selected":false,"positionAbsolute":{"x":-468.2721754074064,"y":-1095.4132186666675}},{"width":172,"height":74,"id":"easingNode_bdjNf","data":{"label":"easingNode_bdjNf","easingFn":"easeInSine","variableName":"easingNode_bdjNf"},"type":"easingNode","dragging":false,"position":{"x":943.7278245925936,"y":-659.5798853333342},"selected":false,"positionAbsolute":{"x":943.7278245925936,"y":-659.5798853333342}},{"width":160,"height":147,"id":"operatorNode_AHegu","data":{"label":"operatorNode_AHegu","operator":3,"variableName":"operatorNode_AHegu","defaultValues":{"operand_1":{"type":"number","value":0},"operand_2":{"type":"number","value":2.5}}},"type":"operatorNode","dragging":false,"position":{"x":1209.191044757203,"y":-1135.375024222222},"selected":false,"positionAbsolute":{"x":1209.191044757203,"y":-1135.375024222222}},{"width":160,"height":66,"id":"valueNode_lZHtu","data":{"max":10,"label":"valueNode_lZHtu","value":2.5,"variableName":"valueNode_lZHtu"},"type":"valueNode","dragging":false,"position":{"x":876.8473418034777,"y":-1060.3605730457055},"selected":false,"positionAbsolute":{"x":876.8473418034777,"y":-1060.3605730457055}}],"edges":[{"id":"reactflow__edge-valueNode_Biisyvalue-operatorNode_nKzQkoperand_2","source":"valueNode_Biisy","target":"operatorNode_nKzQk","selected":false,"sourceHandle":"value","targetHandle":"operand_2"},{"id":"reactflow__edge-valueNode_lYzmYvalue-operatorNode_neoYvoperand_1","source":"valueNode_lYzmY","target":"operatorNode_neoYv","selected":false,"sourceHandle":"value","targetHandle":"operand_1"},{"id":"reactflow__edge-mathNode_UuJygoutput-operatorNode_neoYvoperand_2","source":"mathNode_UuJyg","target":"operatorNode_neoYv","selected":false,"sourceHandle":"output","targetHandle":"operand_2"},{"id":"reactflow__edge-timeNode_cLYzetime-operatorNode_nKzQkoperand_1","source":"timeNode_cLYze","target":"operatorNode_nKzQk","selected":false,"sourceHandle":"time","targetHandle":"operand_1"},{"id":"reactflow__edge-timeNode_jYYGztime-operatorNode_hddluoperand_2","source":"timeNode_jYYGz","target":"operatorNode_hddlu","selected":false,"sourceHandle":"time","targetHandle":"operand_2"},{"id":"reactflow__edge-valueNode_gSXlEvalue-operatorNode_hddluoperand_1","source":"valueNode_gSXlE","target":"operatorNode_hddlu","selected":false,"sourceHandle":"value","targetHandle":"operand_1"},{"id":"reactflow__edge-valueNode_rpCCevalue-mathNode_UuJygmath_fn_input_1","source":"valueNode_rpCCe","target":"mathNode_UuJyg","selected":false,"sourceHandle":"value","targetHandle":"math_fn_input_1"},{"id":"reactflow__edge-easingNode_bqAhhy-mathNode_UuJygmath_fn_input_0","source":"easingNode_bqAhh","target":"mathNode_UuJyg","selected":false,"sourceHandle":"y","targetHandle":"math_fn_input_0"},{"id":"reactflow__edge-valueNode_ARGrYvalue-operatorNode_edAktoperand_2","source":"valueNode_ARGrY","target":"operatorNode_edAkt","selected":false,"sourceHandle":"value","targetHandle":"operand_2"},{"id":"reactflow__edge-operatorNode_edAktoutput-conditionNode_NuykUcondition_a","source":"operatorNode_edAkt","target":"conditionNode_NuykU","selected":false,"sourceHandle":"output","targetHandle":"condition_a"},{"id":"reactflow__edge-valueNode_cPQRAvalue-conditionNode_NuykUcondition_b","source":"valueNode_cPQRA","target":"conditionNode_NuykU","selected":false,"sourceHandle":"value","targetHandle":"condition_b"},{"id":"reactflow__edge-operatorNode_edAktoutput-conditionNode_NuykUcondition_true","source":"operatorNode_edAkt","target":"conditionNode_NuykU","selected":false,"sourceHandle":"output","targetHandle":"condition_true"},{"id":"reactflow__edge-valueNode_cPQRAvalue-conditionNode_NuykUcondition_false","source":"valueNode_cPQRA","target":"conditionNode_NuykU","selected":false,"sourceHandle":"value","targetHandle":"condition_false"},{"id":"reactflow__edge-conditionNode_NuykUoutput-easingNode_bqAhhx","source":"conditionNode_NuykU","target":"easingNode_bqAhh","selected":false,"sourceHandle":"output","targetHandle":"x"},{"id":"reactflow__edge-operatorNode_hddluoutput-operatorNode_edAktoperand_1","source":"operatorNode_hddlu","target":"operatorNode_edAkt","selected":false,"sourceHandle":"output","targetHandle":"operand_1"},{"id":"reactflow__edge-operatorNode_neoYvoutput-easingNode_AIbfmx","source":"operatorNode_neoYv","target":"easingNode_AIbfm","selected":false,"sourceHandle":"output","targetHandle":"x"},{"id":"reactflow__edge-valueNode_ehGvnvalue-operatorNode_dNObCoperand_2","source":"valueNode_ehGvn","target":"operatorNode_dNObC","selected":false,"sourceHandle":"value","targetHandle":"operand_2"},{"id":"reactflow__edge-timeNode_vRvnytime-operatorNode_uiGHNoperand_1","source":"timeNode_vRvny","target":"operatorNode_uiGHN","selected":false,"sourceHandle":"time","targetHandle":"operand_1"},{"id":"reactflow__edge-operatorNode_uiGHNoutput-operatorNode_dNObCoperand_1","source":"operatorNode_uiGHN","target":"operatorNode_dNObC","selected":false,"sourceHandle":"output","targetHandle":"operand_1"},{"id":"reactflow__edge-valueNode_ZlTlmvalue-operatorNode_uiGHNoperand_2","source":"valueNode_ZlTlm","target":"operatorNode_uiGHN","selected":false,"sourceHandle":"value","targetHandle":"operand_2"},{"id":"reactflow__edge-operatorNode_dNObCoutput-conditionNode_gNWNmcondition_a","source":"operatorNode_dNObC","target":"conditionNode_gNWNm","selected":false,"sourceHandle":"output","targetHandle":"condition_a"},{"id":"reactflow__edge-valueNode_rpCCevalue-conditionNode_gNWNmcondition_b","source":"valueNode_rpCCe","target":"conditionNode_gNWNm","selected":false,"sourceHandle":"value","targetHandle":"condition_b"},{"id":"reactflow__edge-easingNode_AIbfmy-conditionNode_gNWNmcondition_false","source":"easingNode_AIbfm","target":"conditionNode_gNWNm","selected":false,"sourceHandle":"y","targetHandle":"condition_false"},{"id":"reactflow__edge-easingNode_nGvYSy-mathNode_TdXuGmath_fn_input_0","source":"easingNode_nGvYS","target":"mathNode_TdXuG","selected":false,"sourceHandle":"y","targetHandle":"math_fn_input_0"},{"id":"reactflow__edge-valueNode_UbOMovalue-mathNode_TdXuGmath_fn_input_1","source":"valueNode_UbOMo","target":"mathNode_TdXuG","selected":false,"sourceHandle":"value","targetHandle":"math_fn_input_1"},{"id":"reactflow__edge-operatorNode_nKzQkoutput-operatorNode_rEaHgoperand_1","source":"operatorNode_nKzQk","target":"operatorNode_rEaHg","selected":false,"sourceHandle":"output","targetHandle":"operand_1"},{"id":"reactflow__edge-valueNode_FSQaPvalue-operatorNode_rEaHgoperand_2","source":"valueNode_FSQaP","target":"operatorNode_rEaHg","selected":false,"sourceHandle":"value","targetHandle":"operand_2"},{"id":"reactflow__edge-operatorNode_rEaHgoutput-conditionNode_pHuSncondition_a","source":"operatorNode_rEaHg","target":"conditionNode_pHuSn","selected":false,"sourceHandle":"output","targetHandle":"condition_a"},{"id":"reactflow__edge-valueNode_TpLshvalue-conditionNode_pHuSncondition_b","source":"valueNode_TpLsh","target":"conditionNode_pHuSn","selected":false,"sourceHandle":"value","targetHandle":"condition_b"},{"id":"reactflow__edge-operatorNode_rEaHgoutput-conditionNode_pHuSncondition_true","source":"operatorNode_rEaHg","target":"conditionNode_pHuSn","selected":false,"sourceHandle":"output","targetHandle":"condition_true"},{"id":"reactflow__edge-valueNode_TpLshvalue-conditionNode_pHuSncondition_false","source":"valueNode_TpLsh","target":"conditionNode_pHuSn","selected":false,"sourceHandle":"value","targetHandle":"condition_false"},{"id":"reactflow__edge-conditionNode_pHuSnoutput-easingNode_nGvYSx","source":"conditionNode_pHuSn","target":"easingNode_nGvYS","selected":false,"sourceHandle":"output","targetHandle":"x"},{"id":"reactflow__edge-valueNode_TpLshvalue-operatorNode_oeTEroperand_1","source":"valueNode_TpLsh","target":"operatorNode_oeTEr","selected":false,"sourceHandle":"value","targetHandle":"operand_1"},{"id":"reactflow__edge-mathNode_TdXuGoutput-operatorNode_oeTEroperand_2","source":"mathNode_TdXuG","target":"operatorNode_oeTEr","selected":false,"sourceHandle":"output","targetHandle":"operand_2"},{"id":"reactflow__edge-operatorNode_oeTEroutput-conditionNode_gNWNmcondition_true","source":"operatorNode_oeTEr","target":"conditionNode_gNWNm","selected":false,"sourceHandle":"output","targetHandle":"condition_true"},{"id":"reactflow__edge-conditionNode_gNWNmoutput-easingNode_bdjNfx","source":"conditionNode_gNWNm","target":"easingNode_bdjNf","selected":false,"sourceHandle":"output","targetHandle":"x"},{"id":"reactflow__edge-easingNode_bdjNfy-modelBlendshapesOutput_fnNmsmodel_blendshapes_EyeBlinkL","source":"easingNode_bdjNf","target":"modelBlendshapesOutput_fnNms","selected":false,"sourceHandle":"y","targetHandle":"model_blendshapes_EyeBlinkL"},{"id":"reactflow__edge-easingNode_bdjNfy-modelBlendshapesOutput_fnNmsmodel_blendshapes_EyeBlinkR","source":"easingNode_bdjNf","target":"modelBlendshapesOutput_fnNms","selected":false,"sourceHandle":"y","targetHandle":"model_blendshapes_EyeBlinkR"},{"id":"reactflow__edge-easingNode_bdjNfy-operatorNode_AHeguoperand_1","source":"easingNode_bdjNf","target":"operatorNode_AHegu","selected":false,"sourceHandle":"y","targetHandle":"operand_1"},{"id":"reactflow__edge-valueNode_lZHtuvalue-operatorNode_AHeguoperand_2","source":"valueNode_lZHtu","target":"operatorNode_AHegu","selected":false,"sourceHandle":"value","targetHandle":"operand_2"},{"id":"reactflow__edge-operatorNode_AHeguoutput-modelBlendshapesOutput_fnNmsmodel_blendshapes_BrowY","source":"operatorNode_AHegu","target":"modelBlendshapesOutput_fnNms","selected":false,"sourceHandle":"output","targetHandle":"model_blendshapes_BrowY"},{"id":"reactflow__edge-operatorNode_AHeguoutput-modelBlendshapesOutput_fnNmsmodel_blendshapes_ParamBrowRY","source":"operatorNode_AHegu","target":"modelBlendshapesOutput_fnNms","selected":false,"sourceHandle":"output","targetHandle":"model_blendshapes_ParamBrowRY"},{"id":"reactflow__edge-operatorNode_AHeguoutput-modelBlendshapesOutput_fnNmsmodel_blendshapes_ParamBrowLY","source":"operatorNode_AHegu","target":"modelBlendshapesOutput_fnNms","selected":false,"sourceHandle":"output","targetHandle":"model_blendshapes_ParamBrowLY"},{"id":"reactflow__edge-easingNode_bdjNfy-modelBlendshapesOutput_fnNmsmodel_blendshapes_ParamEyeLClose","source":"easingNode_bdjNf","target":"modelBlendshapesOutput_fnNms","selected":false,"sourceHandle":"y","targetHandle":"model_blendshapes_ParamEyeLClose"},{"id":"reactflow__edge-easingNode_bdjNfy-modelBlendshapesOutput_fnNmsmodel_blendshapes_ParamEyeRClose","source":"easingNode_bdjNf","target":"modelBlendshapesOutput_fnNms","selected":false,"sourceHandle":"y","targetHandle":"model_blendshapes_ParamEyeRClose"},{"source":"easingNode_bdjNf","sourceHandle":"y","target":"modelBlendshapesOutput_fnNms","targetHandle":"model_blendshapes_EyeBlinkLeft","id":"reactflow__edge-easingNode_bdjNfy-modelBlendshapesOutput_fnNmsmodel_blendshapes_EyeBlinkLeft"},{"source":"easingNode_bdjNf","sourceHandle":"y","target":"modelBlendshapesOutput_fnNms","targetHandle":"model_blendshapes_EyeBlinkRight","id":"reactflow__edge-easingNode_bdjNfy-modelBlendshapesOutput_fnNmsmodel_blendshapes_EyeBlinkRight"}],"viewport":{"x":811.5052479940873,"y":956.0758053219338,"zoom":0.6399362069221536}}',
          }),
          '*'
        )

        console.log(evt.detail.output.gltfFilename)
      }
    })

    window.addEventListener(
      'keydown',
      (event) => {
        if (event.key === 'Escape') {
          event.preventDefault();
          showImageEditor.val = false;
        }
      },
      {
        capture: true,
      },
    );

    graphCanvas.addEventListener('keydown', (event) => {
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

      if (event.key === 'v') {
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
      // nodeData.input.required.upload = ['IMAGEUPLOAD'];
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

app.registerExtension(ext);
