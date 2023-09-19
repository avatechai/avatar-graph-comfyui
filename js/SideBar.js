import { updateImagePrompts } from "./ImageEditor.js";
import {
  imagePrompts,
  selectedLayer,
  imagePromptsMulti,
  targetNode,
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

export function SideBar() {
  const layer_to_delete = van.state("");

  return div(
    {
      class:
        "ml-2 z-100 w-fit flex-col flex justify-center absolute top-0 left-0 bottom-0 items-start gap-2",
    },

    () => {
      const layers = Object.entries(imagePromptsMulti.val);
      return ul(
        {
          class: "menu bg-base-200 w-56 rounded-box text-base-content",
        },
        layers.length === 0 ? li(a("Empty layer")) : null,
        ...layers.map(([key, value]) => {
          return li(
            a(
              {
                class: () =>
                  `capitalize text-start items-start flex items-center justify-between  ${
                    selectedLayer.val === key ? "active" : ""
                  }`,
                onclick: () => {
                  selectedLayer.val = key;
                  imagePrompts.val = imagePromptsMulti.val[key];
                },
              },
              key,
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
            const outputIndex = targetNode.val.findOutputSlot(
              layer_to_delete.val
            );
            targetNode.val.removeOutput(outputIndex);
            targetNode.val.graph.change();
            updateImagePrompts();
            delete_layer_dialog.close();
          },
        },
        p("Are you sure you want to delete this layer?")
      ),
    dialog(
      { id: "my_modal_3", class: "modal" },
      div(
        { class: "modal-box text-base-content" },
        form(
          {
            class: "gap-2 flex flex-col",
            method: "dialog",
            onsubmit: (e) => {
              e.preventDefault();
              const inputText = e.target.elements[1].value;
              imagePromptsMulti.val = {
                ...imagePromptsMulti.val,
                [inputText]: [],
              };
              console.log(inputText, imagePromptsMulti.val);
              my_modal_3.close();
              e.target.elements[1].value = "";
              targetNode.val.addOutput(inputText, "SAM_PROMPT");
              targetNode.val.graph.change();
              if (
                selectedLayer.val === null ||
                selectedLayer.val === undefined ||
                selectedLayer.val === ""
              ) {
                selectedLayer.val = inputText;
              }
              updateImagePrompts();
            },
          },
          button(
            {
              type: "button",
              class: "btn btn-sm btn-circle btn-ghost absolute right-2 top-2",
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
            class: "btn btn-sm  btn-ghost place-self-end",
          },
          "Confirm"
        )
      )
    )
  );
}
