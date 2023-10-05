import { van } from "./van.js";
const { button, iframe, div, img } = van.tags;
import { showEditor, previewUrl, showPreview } from "./state.js";

export function AvatarPreview() {

  return div(
    { class: () => (!showEditor.val ? "" : "hidden") },
    iframe({
      id: "avatech-viewer-iframe",
      title: "avatech-viewer-iframe",
      name: "avatech-viewer-iframe",
      allow: "cross-origin-isolated",
      class: () =>
        "w-[320px] h-[370px] absolute right-0 top-0 z-[100] pointer-events-auto flex mt-4 mr-4 rounded-2xl border-none " +
        (showPreview.val ? "" : "hidden"),
      // src: "https://labs.avatech.ai/viewer/default",
      // src: "http://localhost:3000/viewer/default",
      src: previewUrl,
    })
  );
}
