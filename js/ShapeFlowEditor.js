import { iframeSrc, showEditor } from "./state.js";
import { van } from "./van.js";
const { button, iframe, div, img, span } = van.tags;

export function ShapeFlowEditor() {
  return div(
    {
      class: "w-full h-full relative",
    },
    button(
      {
        class: () =>
          "ml-2 mt-2 btn btn-circle flex flex-row btn-ghost normal-case absolute px-4 rounded-md left-0 top-0 z-[200] w-fit pointer-events-auto bg-black " +
          (showEditor.val ? "" : "hidden"),
        onclick: () => {
          console.log("close");
          const editor = document.getElementById("avatech-editor-iframe");
          editor.contentWindow.postMessage(
            {
              method: "back",
            },
            "*"
          );
          showEditor.val = false;
        },
      },
      span({
        class: "iconify text-lg",
        "data-icon": "ic:baseline-arrow-back",
        "data-inline": "false",
      }),
      div("Back & Save")
    ),
    iframe({
      id: "avatech-editor-iframe",
      title: "avatech-editor-iframe",
      name: "avatech-editor-iframe",
      allow: "cross-origin-isolated; clipboard-read; clipboard-write",
      class: () =>
        "w-full h-full pointer-events-auto " + (showEditor.val ? "" : "hidden"),
      src: iframeSrc,
    })
  );
}
