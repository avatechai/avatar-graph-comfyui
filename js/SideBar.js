import { drawSegment, getClicks, updateImagePrompts } from "./LayerEditor.js";
import {
  imagePrompts,
  selectedLayer,
  imagePromptsMulti,
  targetNode,
  showImageEditor,
  allImagePrompts,
  samPrompts,
} from "./state.js";
import { van } from "./van.js";
const {
  button,
  iframe,
  div,
  img,
  h3,
  p,
  form,
  dialog,
  input,
  li,
  ul,
  a,
  span,
} = van.tags;

export const updateOutputs = () => {
  const outputNames = targetNode.val.outputs.map((x) => x.name).slice(1);
  const record = Object.keys(imagePromptsMulti.val);

  const diff = outputNames.filter((x) => !record.includes(x));
  const missingDiff = record.filter((x) => !outputNames.includes(x));

  if (diff.length > 0) {
    console.log("Cleaning up missing output slots", diff);
    diff.forEach((x) => {
      targetNode.val.removeOutput(targetNode.val.findOutputSlot(x));
    });
    targetNode.val.graph.change();
  }

  if (missingDiff.length > 0) {
    console.log("Adding missing output slots", diff);
    missingDiff.forEach((x) => {
      targetNode.val.addOutput(
        x,
        targetNode.val.type === "SAM MultiLayer" ? "IMAGE" : "SAM_PROMPT"
      );
    });
    targetNode.val.graph.change();
  }
};

van.derive(() => {
  if (
    showImageEditor.val &&
    targetNode.val != undefined &&
    targetNode.val.outputs &&
    targetNode.val.type === "SAM MultiLayer"
  ) {
    updateOutputs();
  }
});

export function SideBar() {
  const layer_to_delete = van.state("");

  return div(
    div(
      {
        class:
          "ml-2 z-100 w-fit flex-col flex justify-center absolute top-0 left-0 bottom-0 items-start gap-2",
      },
      () => {
        const layers = Object.entries(imagePromptsMulti.val);
        return ul(
          {
            class: "menu bg-base-200 w-56 rounded-box text-base-content ",
          },
          button(
            {
              onclick: () => {
                layers.map(([key, value]) => {
                  imagePrompts.val = [];
                  imagePromptsMulti.val[key] = [];
                });
                drawSegment([]);
                updateImagePrompts();
              },
              class: "btn btn-ghost normal-case flex",
            },
            "Clear ALL"
          ),
          layers.length === 0 ? li(a("Empty layer")) : null,
          ...layers.map(([key, value]) => {
            return li(
              a(
                {
                  class: () =>
                    `normal-case text-start items-start flex items-center justify-between  ${
                      selectedLayer.val === key ? "active" : ""
                    }`,
                  onclick: () => {
                    selectedLayer.val = key;
                    imagePrompts.val = imagePromptsMulti.val[key];
                    drawSegment(getClicks());
                  },
                },
                key,
                div(
                  {},
                  button(
                    {
                      class:
                        "btn btn-circle btn-xs btn-ghost group hover:text-red-500",
                      onclick: (e) => {
                        console.log("clear");
                        e.preventDefault();
                        e.stopPropagation();
                        imagePrompts.val = [];
                        imagePromptsMulti.val[key] = [];
                        drawSegment([]);
                        updateImagePrompts();
                      },
                    },
                    span({
                      class: "iconify",
                      "data-icon": "ant-design:clear-outlined",
                      "data-inline": "false",
                    })
                  ),
                  button(
                    {
                      class:
                        "btn btn-circle btn-xs btn-ghost group hover:text-red-500",
                      onclick: (e) => {
                        console.log("delete");
                        e.preventDefault();
                        e.stopPropagation();
                        layer_to_delete.val = key;
                        setTimeout(() => {
                          delete_layer_dialog.showModal();
                        }, 0);
                      },
                    },
                    span({
                      class: "iconify",
                      "data-icon": "ic:baseline-delete",
                      "data-inline": "false",
                    })
                  )
                )
              )
            );
          }),
          div({ class: "divider !py-0 my-0" }),
          li(
            a(
              {
                class: "flex items-center justify-between",
                onclick: () => {
                  my_modal_3.showModal();
                },
              },
              "New Layer",
              span({
                class: "iconify",
                "data-icon": "ic:outline-plus",
                "data-inline": "false",
              })
            )
          )
        );
      },
      () =>
        ConfirmDialog(
          {
            id: "delete_layer_dialog",
            title: "Delete Layer: " + layer_to_delete.val,
            onsubmit: () => {
              imagePromptsMulti.val = Object.fromEntries(
                Object.entries(imagePromptsMulti.val).filter(
                  ([key, value]) => key !== layer_to_delete.val
                )
              );
              console.log(imagePromptsMulti.val);
              if (selectedLayer.val === layer_to_delete.val) {
                // Select another layer if there is one
                if (Object.keys(imagePromptsMulti.val).length > 0) {
                  selectedLayer.val = Object.keys(imagePromptsMulti.val)[0];
                  imagePrompts.val = imagePromptsMulti.val[selectedLayer.val];
                } else {
                  selectedLayer.val = "";
                  imagePrompts.val = [];
                }
              }
              targetNode.val.graph.change();
              updateImagePrompts();
              delete_layer_dialog.close();
            },
          },
          p("Are you sure you want to delete this layer?")
        ),
      () =>
        dialog(
          { id: "my_modal_3", class: "modal" },
          div(
            { class: "modal-box text-base-content" },
            form(
              {
                class: "gap-2 flex flex-col",
                method: "dialog",
                onsubmit: (e) => {
                  console.log("add new layer");
                  e.preventDefault();
                  const inputText = e.target.elements[1].value;
                  imagePromptsMulti.val = {
                    ...imagePromptsMulti.val,
                    [inputText]: [],
                  };
                  console.log(inputText, imagePromptsMulti.val);
                  my_modal_3.close();
                  e.target.elements[1].value = "";

                  selectedLayer.val = inputText;
                  imagePrompts.val = imagePromptsMulti.val[inputText];
                  drawSegment(getClicks());
                  updateImagePrompts();
                },
              },
              button(
                {
                  type: "button",
                  class:
                    "btn btn-sm btn-circle btn-ghost absolute right-2 top-2",
                  onclick: (e) => {
                    e.stopPropagation();
                    my_modal_3.close();
                  },
                },
                "✕"
              ),
              h3(
                { class: "font-bold text-lg text-base-content" },
                "Add new layer!"
              ),
              input({
                type: "text",
                placeholder: "Type here",
                class: "input input-bordered w-full",
                autofocus: true,
              }),
              button(
                {
                  type: "submit",
                  class: "btn btn-sm  btn-ghost place-self-end",
                },
                "Confirm"
              )
            )
          )
        )
    ),
    div(
      {
        class:
          "ml-2 z-100 w-fit flex-col flex justify-center absolute top-0 right-0 bottom-0 items-start gap-2 bg-transparent",
      },
      () => {
        return ul(
          {
            class: "menu bg-base-200 w-56 rounded-box text-base-content ",
          },
          span("Segment History"),
          ...allImagePrompts.val.map((e) =>
            li(
              a(
                {
                  class:
                    "normal-case text-start flex items-center justify-between",
                  onclick: async () => {
                    imagePromptsMulti.val = e.prompt;
                    imagePrompts.val = imagePromptsMulti.val[selectedLayer.val];
                    drawSegment(getClicks());
                    updateImagePrompts();
                  },
                },
                e.version
              )
            )
          )
        );
      }
    )
  );
}

function ConfirmDialog({ id, title, onsubmit }, ...children) {
  return dialog(
    { id: id, class: "modal" },
    div(
      { class: "modal-box text-base-content" },
      form(
        {
          class: "gap-2 flex flex-col",
          method: "dialog",
          onsubmit: onsubmit,
        },
        button(
          {
            type: "button",
            class: "btn btn-sm btn-circle btn-ghost absolute right-2 top-2",
            onclick: (e) => {
              e.stopPropagation();
              window[id].close();
            },
          },
          "✕"
        ),
        h3({ class: "font-bold text-lg text-base-content" }, title),
        ...children,
        button(
          {
            type: "submit",
            autofocus: true,
            class: "btn btn-sm  btn-ghost place-self-end",
          },
          "Confirm"
        )
      )
    )
  );
}
