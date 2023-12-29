import { combinePointsNode, samPrompts } from "./state.js";
import { van } from "./van.js";
const { div, dialog, form, button, h3, input, span } = van.tags;

van.derive(() => {
  if (
    combinePointsNode.val != undefined &&
    combinePointsNode.val.type === "Combine Points"
  ) {
    const inputNames =
      combinePointsNode.val.inputs?.map((x) => x.name).slice(1) || [];
    const record = Object.keys(samPrompts.val);

    const missingDiff = record.filter((x) => !inputNames.includes(x));

    if (missingDiff.length > 0) {
      missingDiff.forEach((x) => {
        combinePointsNode.val.addInput(x, "POINTS");
      });
      combinePointsNode.val.graph.change();
    }
  }
});

export function CombinePointsDialog() {
  const showAddLayer = van.state(false);

  return div(
    {
      class: () =>
        "absolute z-[100] top-0 left-0 flex justify-center w-full h-full ",
    },
    () =>
      dialog(
        { id: "combine_points_dialog", class: "modal" },
        div(
          { class: "modal-box text-base-content" },
          form(
            {
              class: "gap-2 flex flex-col",
              method: "dialog",
              onsubmit: (e) => {
                e.preventDefault();
                combine_points_dialog.close();
              },
            },
            button(
              {
                type: "button",
                class: "btn btn-sm btn-circle btn-ghost absolute right-2 top-2",
                onclick: (e) => {
                  e.stopPropagation();
                  combine_points_dialog.close();
                },
              },
              "✕"
            ),
            h3({ class: "font-bold text-lg text-base-content" }, "Edit points"),
            () =>
              div(
                { class: "flex flex-col gap-2 mb-2" },
                ...Object.keys(samPrompts.val).map((key) => {
                  return span(key);
                })
              ),

            () =>
              showAddLayer.val
                ? div(
                    {
                      class:
                        "flex flex-row justify-center items-center border rounded-md pr-2",
                    },
                    input({
                      type: "text",
                      placeholder: "Type here",
                      id: "layerName",
                      class:
                        "input input-ghost w-full focus:ring-0 focus:border-none focus:outline-none",
                      autofocus: true,
                    }),
                    button(
                      {
                        onclick: (e) => {
                          e.stopPropagation();
                          e.preventDefault();
                          showAddLayer.val = false;
                        },
                      },
                      span({
                        class: "iconify text-2xl",
                        "data-icon": "iconoir:cancel",
                      })
                    ),
                    button(
                      {
                        onclick: (e) => {
                          e.stopPropagation();
                          e.preventDefault();
                          showAddLayer.val = false;
                          const inputText =
                            document.getElementById("layerName").value;
                          samPrompts.val = {
                            ...samPrompts.val,
                            [inputText]: [],
                          };
                        },
                      },
                      span({
                        class: "iconify text-2xl",
                        "data-icon": "mdi:tick",
                      })
                    )
                  )
                : button(
                    {
                      class: "btn btn-outline btn",
                      onclick: (e) => {
                        e.stopPropagation();
                        e.preventDefault();
                        showAddLayer.val = true;
                      },
                    },
                    "Add new layer"
                  ),
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
