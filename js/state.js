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
 * @property {>} label - The label
 *
 * @typedef {Object} Box
 * @property {number} x1
 * @property {number} y1
 * @property {number} x2
 * @property {number} y2
 */

import { van } from "./van.js";
export const iframeSrc = van.state("https://editor.avatech.ai?comfyui=true");
export const showEditor = van.state(false);
// localStorage.getItem("showPreview") == 'true'
console.log(localStorage.getItem("showPreview"));
if (localStorage.getItem("showPreview") == null)
  localStorage.setItem("showPreview", 'true')
export const showPreview = van.state(localStorage.getItem("showPreview") == 'true');
export const previewUrl = van.state(
  "https://editor.avatech.ai/viewer?avatarId=default&debug=true&width=350&height=350&hideTrigger=true&voiceSelection=true&hideUI=true"
);
export const previewImg = van.state("");
export const previewImgLoading = van.state(false);
export const enableAutoSegment = van.state(false);
// export const previewUrl = van.state("http://localhost:3006/viewer?avatarId=default&hideUI=true&debug=true&width=300&height=300&showAudioControl=true");
export const isDirty = van.state(false);
export const fileName = van.state("");
export const showImageEditor = van.state(false);
export const showLoading = van.state(false);
export const alertDialog = van.state({
  text: "",
  time: 0,
});
export const shareLoading = van.state(false);
export const previewModelId = van.state("");

export const isGenerateFlow = van.state(false);

export const loadingCaption = van.state("");
export const imageUrl = van.state("");
export const point_label = van.state(1);
export const imageContainerSize = van.state({
  width: 0,
  height: 0,
});

/** @type {State<Box>} */
export const boxes = van.state();

/** @type {State<Record<string, Box>>} */
export const boxesMulti = van.state({});

/** @type {State<Point[]>} */
export const imagePrompts = van.state([]);

export const allImagePrompts = van.state([{}]);

/** @type {State<Record<string, Point[]>>} */
export const imagePromptsMulti = van.state({});

/** @type {State<string>} */
export const selectedLayer = van.state();

/** @type {State<LGraphNode>} */
export const targetNode = van.state();

export const imageSize = van.state({ width: 0, height: 0, samScale: 0 });
export const embeddings = van.state();
export const embeddingID = van.state("Test");

/** @type {State<LGraphNode>} */
export const combinePointsNode = van.state();
export const samPrompts = van.state({});
