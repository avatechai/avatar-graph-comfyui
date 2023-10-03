import { van } from "./van.js";
const { button, iframe, div, img } = van.tags;
import { showEditor, previewUrl } from "./state.js";

export function AvatarPreview() {
  return div(
    {
      class: () =>
        "w-[320px] h-[370px] absolute right-0 top-0 z-[100] pointer-events-auto mt-4 mr-4 " +
        (!showEditor.val ? "" : "hidden"),
    },
    iframe({
      id: "avatech-viewer-iframe",
      title: "avatech-viewer-iframe",
      name: "avatech-viewer-iframe",
      allow: "cross-origin-isolated",
      class: () =>
        "w-full h-full flex pointer-events-auto rounded-2xl border-none " +
        (!showEditor.val ? "" : "hidden"),
      // src: "https://labs.avatech.ai/viewer/default",
      // src: "http://localhost:3000/viewer/default",
      src: previewUrl,
    }),
  );
}
