import { imagePrompts, selectedLayer, imagePromptsMulti } from "./state.js";
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
  return div(
    {
      class:
        "z-100 w-fit flex-col flex justify-center absolute top-0 left-0 bottom-0 items-start gap-2",
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
                  `capitalize text-start items-start ${
                    selectedLayer.val === key ? "active" : ""
                  }`,
                onclick: () => {
                  selectedLayer.val = key;
                  imagePrompts.val = imagePromptsMulti.val[key];
                },
              },
              key
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
    dialog(
      { id: "my_modal_3", class: "modal" },
      div(
        { class: "modal-box text-base-content" },
        form(
          {
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
            class: "input input-bordered w-full max-w-xs",
            autofocus: true,
          })
        )
      )
    )
  );
}
