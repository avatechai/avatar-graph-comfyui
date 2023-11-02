/**
 * @typedef {import('./van-1.1.3.min.js').State<T>} State<T>
 * @template T
 */

/**
 * @typedef {import('../../../web/types/litegraph.js').LGraphNode} LGraphNode
 *
 * @typedef {Object} Point
 * @property {number} x - The x coordinate
 * @property {number} y - The y coordinate
 * @property {number} label - The label
 */

import { van } from "./van.js";

export const iframeSrc = van.state("https://editor.avatech.ai?comfyui=true");
export const showEditor = van.state(false);
export const showPreview = van.state(localStorage.getItem("showPreview") == 'true');
export const previewUrl = van.state("https://editor.avatech.ai/viewer?avatarId=default&hideUI=true&debug=true&width=300&height=300&showAudioControl=true&hideTrigger=true");
// export const previewUrl = van.state("http://localhost:3006/viewer?avatarId=default&hideUI=true&debug=true&width=300&height=300&showAudioControl=true");
export const isDirty = van.state(false);
export const fileName = van.state('');
export const showImageEditor = van.state(false);
export const showLoading = van.state(false);
export const alertDialog = van.state({
  text: "",
  time: 0,
});
export const shareLoading = van.state(false);
export const previewModelId = van.state('');

export const loadingCaption = van.state("");
export const imageUrl = van.state("");
export const point_label = van.state(1);
export const imageContainerSize = van.state({
  width: 0,
  height: 0,
});

/** @type {State<Point[]>} */
export const imagePrompts = van.state([]);

/** @type {State<Record<string, Point[]>>} */
export const imagePromptsMulti = van.state({});

/** @type {State<string>} */
export const selectedLayer = van.state();

/** @type {State<LGraphNode>} */
export const targetNode = van.state();

export const imageSize = van.state({ width: 0, height: 0, samScale: 0 });
export const embeddings = van.state();
