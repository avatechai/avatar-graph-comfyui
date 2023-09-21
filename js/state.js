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

import { van } from './van.js';

export const iframeSrc = van.state('https://editor.avatech.ai');
export const showEditor = van.state(false);
export const fileName = van.state('');
export const showImageEditor = van.state(false);
export const showLoading = van.state(false);
export const imageUrl = van.state('');
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
