import { van } from "./van.js";
const { button, iframe, div, img } = van.tags;

export function AvatarViewer() {
  return div(
    {
      class:
        "w-[310px] h-[310px] absolute right-0 top-0 z-[100] pointer-events-auto mt-4 mr-4",
    },
    iframe({
      id: "avatech-viewer-iframe",
      title: "avatech-viewer-iframe",
      name: "avatech-viewer-iframe",
      allow: "cross-origin-isolated",
      sandbox: "true",
      class: () =>
        "w-full h-full flex pointer-events-auto rounded-2xl border-none",
      // src: "https://labs.avatech.ai/viewer/default",
      src: 'http://localhost:3000/viewer/default',
    })
  );
}
