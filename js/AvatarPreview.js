import { van } from "./van.js";
import { imageUrl, showPreview, previewUrl, showEditor } from "./state.js";
const { button, iframe, div, img, input, label } = van.tags;
import { app } from "./app.js";

export function AvatarPreview() {
  return div(
    {
      class: () => {
        console.log(showPreview);

        return (
          (showPreview.val ? "" : "hidden ") +
          "w-full h-screen absolute left-0 top-0 z-[100] pointer-events-auto flex border-none "
        );
      },
    },
    div(
      {
        class: "absolute top-4 left-4 flex flex-col gap-2",
      },
      button(
        {
          class: () => " btn btn-outline w-24 text-xs !px-0 normal-case",
          onclick: () => {
            // console.log(showEditor);
            showPreview.val = false;
          },
        },
        "X"
      ),
      button(
        {
          class: () => "btn btn-outline w-24 text-xs !px-0 normal-case",
          onclick: () => document.getElementById("comfy-load-button").click(),
        },
        "Load Shape flow"
      )
    ),
    button(
      {
        class: () =>
          "absolute top-4 right-4 btn text-black btn-ghost w-24 text-xs !px-0 normal-case",
        onclick: () => window.open("https://twitter.com/avatech_gg", "_blank"),
      },
      "Twitter"
    ),
    div(
      {
        class: () =>
          "flex w-full h-full bg-white justify-around items-center p-24" +
          (showPreview.val ? "" : "hidden"),
      },
      div(
        {
          class: () => "flex flex-col bg-white justify-start h-full",
        },
        div(
          {
            class: () =>
              " bg-gradient-to-b from-black via-[#5F5F5F] via-60% to-white text-transparent bg-clip-text font-gabarito text-4xl",
          },
          "Avatech v1"
        ),
        div(
          {
            class: () =>
              " bg-gradient-to-b from-black via-[#5F5F5F] via-50% to-white text-transparent bg-clip-text font-gabarito text-2xl",
          },
          "Get your DALLE3 AI Personal Clone"
        ),
        label(
          {
            class: () =>
              " aspect-square w-full ring-1 ring-black bg-[#f6f6f6] rounded-lg flex justify-center items-center text-black cursor-pointer mt-24",
            // for: "imgtosam",
            onclick: () => {
              /** @type {import('../../../web/types/litegraph.js').LGraph}*/
              const graph = app.graph;
              const nodes = graph.findNodesByType("LoadImage");

              /** @type {any[]}*/
              const widgets = nodes[0].widgets
              console.log(nodes[0]);
              console.log(nodes[0].widgets);
              widgets.find(x => x.value == "image").callback()
            },
          },
          div(
            {
              class: () => " absolute flex justify-center items-center ",
            },
            "upload your image"
          ),
          img({
            class: () => " z-[10] object-contain w-[550px] h-[394px]",
            id: "imgUrl",
          })
        ),
        div(
          {
            class: () => " w-full flex flex-col justify-center items-center",
          },
          button(
            {
              class: () => "btn btn-outline mt-24 w-96 normal-case",
              onclick: () => {
                document.getElementById("sam").click();
              },
            },
            "Edit segment"
          ),
          button(
            {
              class: () => "btn btn-outline mt-4 w-96 normal-case",
              onclick: () => {
                document.getElementById("queue-button").click();
              },
            },
            "Generate"
          )
        )
      ),
      iframe({
        id: "avatech-viewer-iframe",
        title: "avatech-viewer-iframe",
        name: "avatech-viewer-iframe",
        allow: "cross-origin-isolated",
        class: () =>
          "w-[370px] h-[320px] z-[100] pointer-events-auto flex  border-none " +
          (showPreview.val ? "" : "hidden"),
        // src: "https://labs.avatech.ai/viewer/default",
        // src: "http://localhost:3000/viewer/default",
        src: previewUrl,
      })
    )
  );
}
